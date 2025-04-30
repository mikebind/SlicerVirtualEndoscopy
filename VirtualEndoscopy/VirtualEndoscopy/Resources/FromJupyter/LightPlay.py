import vtk
import slicer

# Get the renderer for View1
layoutManager = slicer.app.layoutManager()
threeDWidget = layoutManager.threeDWidget(1)  # reversed for some reason in loaded scene
threeDView = threeDWidget.threeDView()
renderer = threeDView.renderWindow().GetRenderers().GetFirstRenderer()

# Remove all existing lights
lights = renderer.GetLights()
lights.InitTraversal()
light = lights.GetNextItem()
while light:
    renderer.RemoveLight(light)
    light = lights.GetNextItem()

# Create a camera and set it as the active camera for the renderer
camera = renderer.GetActiveCamera()

# Create a point light and attach it to the camera
light = vtk.vtkLight()
light.SetLightTypeToCameraLight()  # Attach the light to the camera
light.SetPositional(True)  # Make the light a point source
light.SetConeAngle(90)

# Set the light color (optional)
# light.SetColor(1.0, 0.0, 0.0)  # Red light
# light.SetAmbientColor(0.2, 0.2, 0.2)  # Ambient light color
# light.SetDiffuseColor(1.0, 0.5, 0.5)  # Diffuse light color
# light.SetSpecularColor(1.0, 1.0, 1.0)  # Specular light color

# Add the new light to the renderer
renderer.AddLight(light)

# Render the scene
threeDView.renderWindow().Render()
