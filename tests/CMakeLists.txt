

cmake_minimum_required(VERSION 3.4)

project(Tests CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)



#GLEW
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})
link_directories(${GLEW_LIBRARY_DIRS})
add_definitions(${GLEW_DEFINITIONS})


find_package(glm REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})
link_directories(${GLM_LIBRARY_DIRS})
add_definitions(${GLM_DEFINITIONS})


find_package(Freetype REQUIRED)


find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})



add_executable(main math.cpp)
target_link_libraries(main ${FREETYPE_LIBRARY} 
${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} )
target_include_directories(main PRIVATE 
${FREETYPE_INCLUDE_DIRS})


