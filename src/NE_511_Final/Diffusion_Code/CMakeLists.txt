
add_executable( Diffusion_Code Diffusion_Code.cpp)
target_link_libraries (Diffusion_Code Eigen3::Eigen)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  target_link_libraries( Diffusion_Code ${Boost_LIBRARIES})
endif()
