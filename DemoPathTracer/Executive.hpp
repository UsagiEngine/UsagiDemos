#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "UsagiCore.hpp"

namespace Usagi
{
// -----------------------------------------------------------------------------
// Usagi Task Graph Execution Host
// -----------------------------------------------------------------------------

/*
 * Shio:
 * TaskGraphExecutionHost acts as the primary scheduler.
 * It builds a Directed Acyclic Graph (DAG) for parallel system execution
 * and provides a deferred task queue for cyclic requests.
 * It also supports data-parallelism via `parallel_for`.
 */
class TaskGraphExecutionHost
{
    struct SystemNode
    {
        std::function<void()> execute_func;
        std::vector<size_t>   write_deps;
        std::vector<size_t>   read_deps;
        std::vector<size_t>   dependents;
        size_t                initial_dependencies    = 0;
        std::atomic<size_t>   unresolved_dependencies { 0 };
    };

    std::deque<SystemNode>   nodes;
    std::vector<std::thread> workers;
    std::atomic<bool>        running { true };

    std::mutex                        task_mutex;
    std::condition_variable           task_cv;
    std::deque<std::function<void()>> task_queue;

    std::atomic<size_t>     completed_nodes { 0 };
    std::mutex              completion_mutex;
    std::condition_variable completion_cv;

    std::mutex                         deferred_mutex;
    std::vector<std::function<void()>> deferred_tasks;

    /*
     * Shio: Unified resource ID generator.
     */
    inline static std::atomic<size_t> next_resource_id { 0 };

    template <typename T>
    static size_t get_resource_id()
    {
        static size_t id = next_resource_id.fetch_add(1);
        return id;
    }

    template <typename... Ts>
    static void extract_ids(ComponentList<Ts...>, std::vector<size_t> & out)
    {
        (out.push_back(get_resource_id<Ts>()), ...);
    }

    template <typename System>
    static void fill_write_deps(std::vector<size_t> & out)
    {
        if constexpr(requires { typename System::WriteComponent; })
        {
            extract_ids(typename System::WriteComponent { }, out);
        }
        if constexpr(requires { typename System::WriteService; })
        {
            extract_ids(typename System::WriteService { }, out);
        }
    }

    template <typename System>
    static void fill_read_deps(std::vector<size_t> & out)
    {
        if constexpr(requires { typename System::ReadComponent; })
        {
            extract_ids(typename System::ReadComponent { }, out);
        }
        if constexpr(requires { typename System::ReadService; })
        {
            extract_ids(typename System::ReadService { }, out);
        }
    }

    void worker_loop()
    {
        while(running)
        {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(task_mutex);
                task_cv.wait(lock, [this] { return !task_queue.empty() || !running; });

                if(!running && task_queue.empty()) return;

                task = std::move(task_queue.front());
                task_queue.pop_front();
            }
            if(task) task();
        }
    }

    void execute_node(size_t node_index)
    {
        nodes[node_index].execute_func();

        for(size_t dep_index : nodes[node_index].dependents)
        {
            if(nodes[dep_index].unresolved_dependencies.fetch_sub(1) == 1)
            {
                std::lock_guard<std::mutex> lock(task_mutex);
                task_queue.push_back([this, dep_index]() { execute_node(dep_index); });
                task_cv.notify_one();
            }
        }

        if(completed_nodes.fetch_add(1) + 1 == nodes.size())
        {
            std::lock_guard<std::mutex> lock(completion_mutex);
            completion_cv.notify_one();
        }
    }

public:
    TaskGraphExecutionHost(size_t thread_count)
    {
        for(size_t i = 0; i < thread_count; ++i)
        {
            workers.emplace_back([this] { worker_loop(); });
        }
    }

    ~TaskGraphExecutionHost()
    {
        running = false;
        task_cv.notify_all();
        for(auto & t : workers)
        {
            if(t.joinable()) t.join();
        }
    }

    template <typename System, typename Entities, typename Services>
    void register_system(System & sys, Entities & entities, Services & services)
    {
        auto & node = nodes.emplace_back();
        node.execute_func = [&sys, &entities, &services]() {
            sys.update(entities, services);
        };
        fill_write_deps<System>(node.write_deps);
        fill_read_deps<System>(node.read_deps);
    }

    void build_graph()
    {
        for(auto & node : nodes)
        {
            node.dependents.clear();
            node.initial_dependencies = 0;
        }

        for(size_t i = 0; i < nodes.size(); ++i)
        {
            for(size_t j = i + 1; j < nodes.size(); ++j)
            {
                bool conflict = false;

                for(size_t j_write : nodes[j].write_deps)
                {
                    if(std::find(nodes[i].write_deps.begin(), nodes[i].write_deps.end(), j_write) != nodes[i].write_deps.end() ||
                       std::find(nodes[i].read_deps.begin(), nodes[i].read_deps.end(), j_write) != nodes[i].read_deps.end())
                    {
                        conflict = true;
                        break;
                    }
                }

                if(!conflict)
                {
                    for(size_t j_read : nodes[j].read_deps)
                    {
                        if(std::find(nodes[i].write_deps.begin(), nodes[i].write_deps.end(), j_read) != nodes[i].write_deps.end())
                        {
                            conflict = true;
                            break;
                        }
                    }
                }

                if(conflict)
                {
                    nodes[i].dependents.push_back(j);
                    nodes[j].initial_dependencies++;
                }
            }
        }
    }

    void submit_deferred_task(std::function<void()> task)
    {
        std::lock_guard<std::mutex> lock(deferred_mutex);
        deferred_tasks.push_back(std::move(task));
    }

    /*
     * Shio: Distributes data-parallel work across the worker thread pool.
     * Blocks the caller until all chunks complete, actively helping to drain the task queue
     * to avoid deadlocks and maximize throughput.
     */
    template <typename F>
    void parallel_for(size_t count, size_t chunk_size, F && func)
    {
        if(count == 0) return;
        
        // Shio: Dynamically override the passed chunk_size to guarantee 100% 32-core occupancy 
        // even when processing extremely small sparse-ray counts!
        size_t hardware_threads = std::max<size_t>(1, std::thread::hardware_concurrency());
        chunk_size = std::max<size_t>(1, count / (hardware_threads * 4));
        
        size_t num_chunks = (count + chunk_size - 1) / chunk_size;

        std::atomic<size_t> next_chunk { 0 };
        size_t pushed_tasks = std::min(num_chunks, workers.size());
        std::atomic<size_t> active_tasks { pushed_tasks };

        auto worker_func = [&]() {
            while(true)
            {
                size_t chunk = next_chunk.fetch_add(1);
                if(chunk >= num_chunks) break;

                size_t start = chunk * chunk_size;
                size_t end   = std::min(start + chunk_size, count);
                func(start, end);
            }
            active_tasks.fetch_sub(1);
        };

        {
            std::lock_guard<std::mutex> lock(task_mutex);
            // Submit tasks up to the number of workers we have
            for(size_t i = 0; i < pushed_tasks; ++i)
            {
                task_queue.push_back(worker_func);
            }
        }
        task_cv.notify_all();

        // The calling thread joins in
        while(true)
        {
            size_t chunk = next_chunk.fetch_add(1);
            if(chunk >= num_chunks) break;

            size_t start = chunk * chunk_size;
            size_t end   = std::min(start + chunk_size, count);
            func(start, end);
        }

        // Wait for all pushed tasks to complete to avoid dangling references
        while(active_tasks.load() > 0)
        {
            std::function<void()> stolen_task;
            {
                std::lock_guard<std::mutex> lock(task_mutex);
                if(!task_queue.empty())
                {
                    stolen_task = std::move(task_queue.front());
                    task_queue.pop_front();
                }
            }
            if(stolen_task) stolen_task();
            else std::this_thread::yield();
        }
    }

    void execute()
    {
        if(nodes.empty()) return;

        completed_nodes = 0;

        {
            std::lock_guard<std::mutex> lock(task_mutex);
            for(size_t i = 0; i < nodes.size(); ++i)
            {
                nodes[i].unresolved_dependencies = nodes[i].initial_dependencies;
                if(nodes[i].initial_dependencies == 0)
                {
                    task_queue.push_back([this, i]() { execute_node(i); });
                }
            }
        }

        task_cv.notify_all();

        {
            std::unique_lock<std::mutex> lock(completion_mutex);
            completion_cv.wait(lock, [this] { return completed_nodes == nodes.size(); });
        }

        while(true)
        {
            std::vector<std::function<void()>> current_deferred;
            {
                std::lock_guard<std::mutex> lock(deferred_mutex);
                if(deferred_tasks.empty()) break;
                current_deferred = std::move(deferred_tasks);
            }

            for(auto & task : current_deferred)
            {
                task();
            }
        }
    }
};
} // namespace Usagi