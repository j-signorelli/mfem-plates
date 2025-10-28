
# Create the executable for linear elasticity
add_executable(panel_le panel_le.cpp)
target_link_libraries(panel_le PUBLIC mfem)

# Create the executable for Kirchoff-Love
add_executable(panel_kl panel_kl.cpp)
target_link_libraries(panel_kl PUBLIC mfem)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/tri_plate.mesh
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

install(TARGETS panel_le DESTINATION bin)
install(TARGETS panel_kl DESTINATION bin)