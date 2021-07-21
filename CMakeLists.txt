cmake_minimum_required(VERSION 3.14)

project(LazyMath
        VERSION 0.0.1
        DESCRIPTION "Header-only library for complex conjugate gradient solver and Levenberg-Marquardt minimizer with contraints"
        HOMEPAGE_URL "https://github.com/tirimatangi/LazyMath"
        LANGUAGES CXX)

# ---- Warning guard ----

# Protect dependents from this project's warnings if the guard isn't disabled
set(LazyMath_warning_guard SYSTEM)
if(LazyMath_INCLUDE_WITHOUT_SYSTEM)
  set(LazyMath_warning_guard "")
endif()

# ---- Declare library ----
# Use release build by default.
if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

if(${CMAKE_BUILD_TYPE} MATCHES "Release")
  message("CMAKE_CXX_FLAGS_RELEASE is ${CMAKE_CXX_FLAGS_RELEASE}")
endif()

if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
  message("CMAKE_CXX_FLAGS_DEBUG is ${CMAKE_CXX_FLAGS_DEBUG}")
endif()


add_library(LazyMath INTERFACE)
add_library(LazyMath::LazyMath ALIAS LazyMath)

target_include_directories(LazyMath
        ${LazyMath_warning_guard}
        INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")

target_compile_features(LazyMath INTERFACE cxx_std_17)

# ---- Install ----

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(LazyMath_directory "LazyMath-${PROJECT_VERSION}")
set(LazyMath_include_directory
        "${CMAKE_INSTALL_INCLUDEDIR}/${LazyMath_directory}")

install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
        DESTINATION "${LazyMath_include_directory}")

install(TARGETS LazyMath
        EXPORT LazyMathTargets
        INCLUDES DESTINATION "${LazyMath_include_directory}")

write_basic_package_version_file(
        LazyMathConfigVersion.cmake
        COMPATIBILITY SameMajorVersion
        ARCH_INDEPENDENT)

set(LazyMath_install_cmakedir
        "${CMAKE_INSTALL_LIBDIR}/cmake/${LazyMath_directory}")

install(FILES
        "${PROJECT_SOURCE_DIR}/cmake/LazyMathConfig.cmake"
        "${PROJECT_BINARY_DIR}/LazyMathConfigVersion.cmake"
        DESTINATION "${LazyMath_install_cmakedir}")

install(EXPORT LazyMathTargets
        NAMESPACE LazyMath::
        DESTINATION "${LazyMath_install_cmakedir}")
