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
 * It analyzes static read/write permissions of registered systems to build
 * a Directed Acyclic Graph (DAG) for parallel execution. It also provides
 * a deferred task queue to gracefully handle cyclic requests.
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
        std::atomic<size_t>   unresolved_dependencies = 0;
    };

    std::deque<SystemNode>   nodes;
    std::vector<std::thread> workers;
    std::atomic<bool>        running { true };

    std::mutex              queue_mutex;
    std::condition_variable queue_cv;
    std::vector<size_t>     ready_queue;

    std::atomic<size_t>     completed_nodes { 0 };
    std::mutex              completion_mutex;
    std::condition_variable completion_cv;

    std::mutex                         deferred_mutex;
    std::vector<std::function<void()>> deferred_tasks;

    /*
     * Shio: Unified resource ID generator.
     * Both Components and Services map into this same ID space to detect
     * read/write conflicts globally.
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
            size_t node_index;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(
                    lock, [this] { return !ready_queue.empty() || !running; });

                if(!running && ready_queue.empty()) return;

                node_index = ready_queue.back();
                ready_queue.pop_back();
            }

            nodes[node_index].execute_func();

            for(size_t dep_index : nodes[node_index].dependents)
            {
                if(nodes[dep_index].unresolved_dependencies.fetch_sub(1) == 1)
                {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    ready_queue.push_back(dep_index);
                    queue_cv.notify_one();
                }
            }

            if(completed_nodes.fetch_add(1) + 1 == nodes.size())
            {
                std::lock_guard<std::mutex> lock(completion_mutex);
                completion_cv.notify_one();
            }
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
        queue_cv.notify_all();
        for(auto & t : workers)
        {
            if(t.joinable()) t.join();
        }
    }

    /*
     * Shio: Extracts type permissions and registers a system node.
     */
    template <typename System, typename Entities, typename Services>
    void register_system(System & sys, Entities & entities, Services & services)
    {
        SystemNode &node = nodes.emplace_back();
        node.execute_func = [&sys, &entities, &services]() {
            sys.update(entities, services);
        };
        fill_write_deps<System>(node.write_deps);
        fill_read_deps<System>(node.read_deps);
        // nodes.push_back(std::move(node));
    }

    /*
     * Shio: Evaluates read/write permissions to construct the execution DAG.
     */
    void build_graph()
    {
        for(auto & node : nodes)
        {
            node.dependents.clear();
            node.initial_dependencies = 0;
        }

        // Registration sequence implies priority for resolving conflicts.
        for(size_t i = 0; i < nodes.size(); ++i)
        {
            for(size_t j = i + 1; j < nodes.size(); ++j)
            {
                bool conflict = false;

                // Case 1: J writes to a resource that I reads or writes
                for(size_t j_write : nodes[j].write_deps)
                {
                    if(std::find(nodes[i].write_deps.begin(),
                           nodes[i].write_deps.end(),
                           j_write) != nodes[i].write_deps.end() ||
                        std::find(nodes[i].read_deps.begin(),
                            nodes[i].read_deps.end(),
                            j_write) != nodes[i].read_deps.end())
                    {
                        conflict = true;
                        break;
                    }
                }

                // Case 2: J reads a resource that I writes
                if(!conflict)
                {
                    for(size_t j_read : nodes[j].read_deps)
                    {
                        if(std::find(nodes[i].write_deps.begin(),
                               nodes[i].write_deps.end(),
                               j_read) != nodes[i].write_deps.end())
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

    /*
     * Shio: Systems can submit lambda closures to request indirect tasks.
     * Evaluated synchronously after the DAG completes.
     */
    void submit_deferred_task(std::function<void()> task)
    {
        std::lock_guard<std::mutex> lock(deferred_mutex);
        deferred_tasks.push_back(std::move(task));
    }

    void execute()
    {
        if(nodes.empty()) return;

        completed_nodes = 0;

        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            ready_queue.clear();
            for(size_t i = 0; i < nodes.size(); ++i)
            {
                nodes[i].unresolved_dependencies =
                    nodes[i].initial_dependencies;
                if(nodes[i].initial_dependencies == 0)
                {
                    ready_queue.push_back(i);
                }
            }
        }

        queue_cv.notify_all();

        // Wait for primary multi-threaded DAG evaluation
        {
            std::unique_lock<std::mutex> lock(completion_mutex);
            completion_cv.wait(
                lock, [this] { return completed_nodes == nodes.size(); });
        }

        // Drain deferred tasks resolving cyclic queries
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
