if(WIN32 AND MSVC)
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION .)

    # HACK: Install Visual Studio's OpenMP library
    if(MSVC11)
        set(MSVC_VER "11")
    elseif(MSVC12)
        set(MSVC_VER "12")
    elseif(MSVC14)
        set(MSVC_VER "14")
    else()
        message(FATAL_ERROR "Unsupported Visual Studio version")
    endif()

    if(CMAKE_CL_64)
        set(CMAKE_MSVC_ARCH x64)
    else()
        set(CMAKE_MSVC_ARCH x86)
    endif()

    include(InstallRequiredSystemLibraries)
    set(OpenMP_REDIST_LIB "${MSVC${MSVC_VER}_REDIST_DIR}/${CMAKE_MSVC_ARCH}/Microsoft.VC${MSVC_VER}0.OpenMP/vcomp${MSVC_VER}0.dll")
    install(PROGRAMS ${OpenMP_REDIST_LIB} DESTINATION ${CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION})
endif()

add_subdirectory(bundle2pset)
add_subdirectory(dmrecon)
add_subdirectory(fssrecon)
add_subdirectory(makescene)
add_subdirectory(mesh2pset)
add_subdirectory(meshalign)
add_subdirectory(meshclean)
add_subdirectory(meshconvert)
add_subdirectory(prebundle)
add_subdirectory(scene2pset)
add_subdirectory(sceneupgrade)
add_subdirectory(sfmrecon)
add_subdirectory(smvsrecon)
add_subdirectory(umve)
