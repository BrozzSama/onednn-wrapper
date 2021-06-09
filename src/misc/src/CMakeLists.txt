find_package(dnnl CONFIG REQUIRED HINTS ${DNNLROOT} ${ONEAPI_CMAKE_CONFIG_HINT})
include_directories(${DNNL_EXAMPLES_SOURCE_DIR}/layers/include)
include_directories(${DNNL_EXAMPLES_SOURCE_DIR}/misc/include)
add_library(misc util.cpp intel_utils.cpp data_loader.cpp)
target_link_libraries(misc DNNL::dnnl)

