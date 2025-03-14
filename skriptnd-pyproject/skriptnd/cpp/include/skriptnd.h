#ifndef _TSC_H_
#define _TSC_H_

#include <set>
#include <optional>
#include "model.h"
#include "error.h"


namespace ts
{
    
    std::string model_name_from_path( const std::string& path );
    std::vector<std::string> enum_graph_names( std::istream& is );

    OperationCallback make_operation_callback( std::set<std::string>&& names );
    
    std::optional<Model> read_model( const std::string& path, const std::string& graph_name, const std::string& stdlib_path,
                                    const ErrorCallback error, const OperationCallback atomic = nullptr, const OperationCallback unroll = nullptr,
                                    const std::map<std::string, ts::ValueExpr>& attribs = {} );
    std::optional<Model> read_model( std::istream& is, const std::string& module, const std::string& graph_name,
                                    const std::string& stdlib_path, const std::string& import_path,
                                    const ErrorCallback error, const OperationCallback atomic = nullptr, const OperationCallback unroll = nullptr,
                                    const std::map<std::string, ts::ValueExpr>& attribs = {} );
    
}   // namespace ts


#endif
