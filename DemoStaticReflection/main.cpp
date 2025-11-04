// #include "CommandLineParser2.hpp"
#include "CatTuple.hpp"
#include "CommandLineParser.hpp"
#include "EnumToString.hpp"
#include "IntegerSequence.hpp"
#include "ListOfSizes.hpp"
#include "MakeNameTuple2.hpp"
#include "MakeNamedTuple.hpp"
#include "StructOfArray.hpp"
#include "StructToTuple.hpp"
#include "TicketCounter.hpp"
#include "Tuple.hpp"
#include "UniversalFormatter.hpp"
#include "Variant.hpp"

template <>
struct std::formatter<R> : universal_formatter
{
};

template <>
struct std::formatter<R2> : universal_formatter
{
};

int main(int argc, const char *argv[])
{
    parse_args(argc, argv);
    print_z();
    std::println("{}", make_named_tuple());
    std::println("{}", make_named_tuple2());
    std::println("{}", cat_tuple_index5());
    std::println("{}", struct_to_tuple());
    // parse_args2(argc, argv); not working
    print_struct_of_arrays();
    print_variants();
    test_tuple();
    print_enum_to_string();
}
