cmake_minimum_required( VERSION 3.2 )
project( SCPT LANGUAGES CXX )

set(CMAKE_CXX_STANDARD 17)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_subdirectory( ${CMAKE_CURRENT_SOURCE_DIR}/src/NE_511_Final/Diffusion_Code/ )
