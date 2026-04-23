from paraview.simple import *
import os

# Load the solutions
# If you get file not found errors, use absolute paths below:
# standard = PVDReader(FileName="/absolute/path/to/torus_diff/standard_solution.pvd")
# anisotropic = PVDReader(FileName="/absolute/path/to/torus_diff/anisotropic_solutions.pvd")
standard = PVDReader(FileName="standard_solution.pvd")
anisotropic = PVDReader(FileName="anisotropic_solutions.pvd")

# Create a layout
layout = GetLayout()
# layout.SetSize(1920, 1080)  # Commented out for compatibility

# Create views
view1 = CreateRenderView()
view1.ViewSize = [960, 1080]
view1.CameraPosition = [0, 0, 5]
view1.CameraFocalPoint = [0, 0, 0]
view1.CameraViewUp = [0, 1, 0]

view2 = CreateRenderView()
view2.ViewSize = [960, 1080]
view2.CameraPosition = [0, 0, 5]
view2.CameraFocalPoint = [0, 0, 0]
view2.CameraViewUp = [0, 1, 0]

# Add views to layout
layout.AddView(view1)
layout.AddView(view2)

# Show data in views
standardDisplay = Show(standard, view1)
anisotropicDisplay = Show(anisotropic, view2)

# Set color map
standardDisplay.SetScalarBarVisibility(view1, True)
anisotropicDisplay.SetScalarBarVisibility(view2, True)

# Create animation
animation = GetAnimationScene()
# You may want to adjust these parameters if needed
animation.NumberOfFrames = 180
animation.StartTime = 0
animation.EndTime = 6

# Save animation
SaveAnimation("solution_animation.mp4", view1, view2, 
             ImageResolution=[1920, 1080],
             FrameRate=30,
             Compression=True)
