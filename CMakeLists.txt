cmake_minimum_required(VERSION 3.16)
project(PerfectMahjongCPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  add_compile_options(/bigobj /utf-8)
else()
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64)")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
  else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  endif()
endif()

find_package(Threads REQUIRED)

if(NOT Torch_DIR AND EXISTS "$ENV{HOME}/.venvs/perfectdou-cpp/lib/python3.13/site-packages/torch/share/cmake/Torch")
  set(Torch_DIR "$ENV{HOME}/.venvs/perfectdou-cpp/lib/python3.13/site-packages/torch/share/cmake/Torch")
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(perfectmahjong STATIC
  src/mahjong_algo/fan_calculator.cpp
  src/mahjong_algo/shanten.cpp
  src/mahjong_algo/stringify.cpp
  src/mcr/tiles.cpp
  src/mcr/oracle.cpp
  src/mcr/game.cpp
  src/mcr/features.cpp
  src/algo/rollout.cpp
  src/algo/ppo.cpp
  src/algo/eval_mcr.cpp
  src/torch_nn/models.cpp
  src/torch_nn/ppo_learner.cpp
  src/torch_nn/tensor_bridge.cpp
)
target_include_directories(perfectmahjong PUBLIC src ${TORCH_INCLUDE_DIRS})
target_link_libraries(perfectmahjong PUBLIC Threads::Threads ${TORCH_LIBRARIES})

if(WIN32)
  target_link_libraries(perfectmahjong PUBLIC ws2_32)
endif()

add_executable(perfectmahjong_train src/train.cpp)
target_link_libraries(perfectmahjong_train PRIVATE perfectmahjong)

add_executable(perfectmahjong_eval src/eval_mcr.cpp)
target_link_libraries(perfectmahjong_eval PRIVATE perfectmahjong)

enable_testing()
include(FetchContent)
if(NOT FETCHCONTENT_SOURCE_DIR_GOOGLETEST AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../PerfectDouCPP/build/_deps/googletest-src")
  set(FETCHCONTENT_SOURCE_DIR_GOOGLETEST "${CMAKE_CURRENT_SOURCE_DIR}/../PerfectDouCPP/build/_deps/googletest-src" CACHE PATH "" FORCE)
endif()
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

add_executable(test_rules tests/test_rules.cpp)
target_link_libraries(test_rules PRIVATE perfectmahjong GTest::gtest_main)
gtest_discover_tests(test_rules)

add_executable(test_algo tests/test_algo.cpp)
target_link_libraries(test_algo PRIVATE perfectmahjong GTest::gtest_main)
gtest_discover_tests(test_algo)

add_executable(test_torch_nn tests/test_torch_nn.cpp)
target_link_libraries(test_torch_nn PRIVATE perfectmahjong GTest::gtest_main)
gtest_discover_tests(test_torch_nn)
