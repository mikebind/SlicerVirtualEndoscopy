import numpy as np
import slicer
from typing import Optional
import pathlib


def get_camera_node_for_View1():
    layoutManager = slicer.app.layoutManager()
    for threeDViewIndex in range(layoutManager.threeDViewCount):
        view = layoutManager.threeDWidget(threeDViewIndex).threeDView()
        threeDViewNode = view.mrmlViewNode()
        viewName = threeDViewNode.GetName()
        if viewName == "View1":
            cameraNode = slicer.modules.cameras.logic().GetViewActiveCameraNode(
                threeDViewNode
            )
            print("View1 camera node ID is: " + cameraNode.GetID())
            return cameraNode


def get_camera_node_for_View2():
    layoutManager = slicer.app.layoutManager()
    for threeDViewIndex in range(layoutManager.threeDViewCount):
        view = layoutManager.threeDWidget(threeDViewIndex).threeDView()
        threeDViewNode = view.mrmlViewNode()
        viewName = threeDViewNode.GetName()
        if viewName == "View2":
            cameraNode = slicer.modules.cameras.logic().GetViewActiveCameraNode(
                threeDViewNode
            )
            print("View2 camera node ID is: " + cameraNode.GetID())
            return cameraNode


def print_cam_params(camNode):
    # For reproducibility, record the location, focal point, and viewUp of the
    # 3D view camera (the other one is manipulated directly, so no need to record)
    print(
        "cam2_position = %s\ncam2_focalPoint = %s\ncam2_viewUp = %s"
        % (
            str(camNode.GetCamera().GetPosition()),
            str(camNode.GetCamera().GetFocalPoint()),
            str(camNode.GetCamera().GetViewUp()),
        )
    )


def make_camera_cone(
    camLoc, focLoc, coneHeight, coneBaseRadius=5.0, modelName="ConeModel"
):
    # Create camera cone, hide it in View1
    coneModelNode = make_cone(
        camLoc, focLoc, coneHeight, coneBaseRadius=coneBaseRadius, modelName=modelName
    )
    dn = coneModelNode.GetDisplayNode()
    dn.SetBackfaceCulling(0)
    dn.SetViewNodeIDs(["vtkMRMLViewNode2"])  # only visible in View2
    return coneModelNode


def get_cone_height(camLoc, focLoc, distFraction=1.0):
    coneHeight = distFraction * np.linalg.norm(np.subtract(camLoc, focLoc))
    return coneHeight


def make_cone(
    tipLoc, openTowardsLoc, coneHeight, coneBaseRadius=5.0, modelName="ConeModel"
):
    """Function to make a cone which starts at a given point, opens towards a second
    given point.  If a modelNode is supplied, that is used, if not, a new model node is
    created and returned.
    """
    import numpy as np

    # Calculate the direction from the base to the tip (not normalized)
    direction = np.subtract(tipLoc, openTowardsLoc)
    distanceFromTipToTowardsLoc = np.linalg.norm(direction)
    # coneHeight = distFraction*distanceFromTipToTowardsLoc
    coneCenter = tipLoc + (coneHeight / 2) * (-direction / distanceFromTipToTowardsLoc)

    # Create vtk cone
    cone = vtk.vtkConeSource()
    cone.SetHeight(coneHeight)
    cone.SetCenter(coneCenter)
    cone.SetDirection(direction)
    cone.SetRadius(coneBaseRadius)
    cone.SetResolution(30)
    cone.SetCapping(0)
    cone.Update()

    # Remove old node of same name if it exists
    try:
        oldModelNode = getNode(modelName)
        slicer.mrmlScene.RemoveNode(oldModelNode)
    except:
        pass

    modelsLogic = slicer.modules.models.logic()
    coneModelNode = modelsLogic.AddModel(cone.GetOutput())
    coneModelNode.SetName(modelName)
    coneModelNode.GetDisplayNode().SetColor(0.54, 0.54, 0.54)

    return coneModelNode


# Replace the endoscopy module... just take a curveNode, resample,
# take screenshot, move camera, take screenshot, etc
# Add StartFrame (most open, curve defined on open frame)


def make_flight_capture(
    curveNode,
    last_loc,
    last_focus,
    camNode,
    coneHeight,
    browserNode,
    mostOpenTimeIdx,
    saveDir,
    filePattern="image_%05d.png",
    pathStepSpacingMm=1,
    focShift=5,
    camViewUp=(0, -0.3, 1),
    coneDynamicOpacity=0.5,
    numFullDynamicLoops=2,
    moveSlices=False,
):
    import ScreenCapture
    import os

    try:
        os.mkdir(saveDir)
        print("Created directory: %s" % saveDir)
    except FileExistsError:
        print("Save directory %s already exists" % saveDir)
    cap = ScreenCapture.ScreenCaptureLogic()
    # Resample curve node to given spacing
    vtkSampledPoints = vtk.vtkPoints()
    slicer.vtkMRMLMarkupsCurveNode.ResamplePoints(
        curveNode.GetCurvePointsWorld(), vtkSampledPoints, pathStepSpacingMm, False
    )
    # Jump to most open time point before flight
    browserNode.SetSelectedItemNumber(mostOpenTimeIdx)
    # Loop over resampled points
    numPoints = vtkSampledPoints.GetNumberOfPoints()
    frameCount = 0
    for pt_ind in range(numPoints):
        loc = vtkSampledPoints.GetPoint(pt_ind)
        # Switch to chosen camera location as last point
        if pt_ind == numPoints - 1:
            loc = last_loc
        # Decide on focal point (either focShift points ahead, or last_focus)
        if pt_ind + focShift >= numPoints - 1:
            foc = last_focus
        else:
            foc = vtkSampledPoints.GetPoint(pt_ind + focShift)
        # Move the camera
        camNode.SetPosition(loc)
        camNode.SetFocalPoint(foc)
        camNode.SetViewUp(camViewUp)
        # Make the camera cone
        coneModelNode = make_camera_cone(loc, foc, coneHeight)
        # Jump slices if requested
        if moveSlices:
            if pt_ind == numPoints - 1:
                # Final location, jump slices to focus
                slicer.modules.markups.logic().JumpSlicesToLocation(
                    *foc, 0
                )  # 0 is for offset, 1 for centered
            else:
                # Show slices at camera location
                slicer.modules.markups.logic().JumpSlicesToLocation(
                    *loc, 0
                )  # 0 is for offset, 1 for centered
        # Capture the screen
        cap.captureImageFromView(
            None, os.path.join(saveDir, (filePattern % frameCount))
        )
        frameCount += 1
    # Modify cone opacity if needed
    coneModelNode.GetDisplayNode().SetOpacity(coneDynamicOpacity)
    # Loop through the browser sequence (to end, then full seq, then full seq again)
    numTimePoints = browserNode.GetNumberOfItems()
    fullTimePointList = list(range(numTimePoints))
    timePointList = list(range(mostOpenTimeIdx + 1, numTimePoints))
    for idx in range(numFullDynamicLoops):
        timePointList += fullTimePointList
    # timePointList = list(range(mostOpenTimeIdx+1,numTimePoints))+list(range(numTimePoints))+list(range(numTimePoints))
    for idx in timePointList:
        browserNode.SetSelectedItemNumber(idx)
        cap.captureImageFromView(
            None, os.path.join(saveDir, (filePattern % frameCount))
        )
        frameCount += 1


def capture_cycles_images(
    browserNode,
    outDir,
    filePattern="tempImage_%05d.png",
    numFullDynamicLoops=2,
    frameStartingIdx=0,
):
    # Loop through the browser sequence (to end, then full seq, then full seq again)
    import ScreenCapture
    import pathlib

    cap = ScreenCapture.ScreenCaptureLogic()
    numTimePoints = browserNode.GetNumberOfItems()
    fullTimePointList = list(range(numTimePoints))
    seqStartIdx = browserNode.GetSelectedItemNumber()
    timePointList = list(range(seqStartIdx + 1, numTimePoints))
    if seqStartIdx > 2:
        numAdditionalRepeats = numFullDynamicLoops
    else:
        numAdditionalRepeats = numFullDynamicLoops - 1
    for idx in range(numAdditionalRepeats):
        timePointList += fullTimePointList
    # timePointList = list(range(mostOpenTimeIdx+1,numTimePoints))+list(range(numTimePoints))+list(range(numTimePoints))
    frameCount = frameStartingIdx
    for idx in timePointList:
        browserNode.SetSelectedItemNumber(idx)
        cap.captureImageFromView(
            None, pathlib.Path.joinpath(outDir, (filePattern % frameCount))
        )
        frameCount += 1


def save_video_from_image_dir(
    imageDir: pathlib.Path,
    imagePattern: str,
    videoFileName: str,
    videoDir: Optional[pathlib.Path] = None,
    videoFrameRateFPS: float = 5,
    videoHeight: int = 1440,
    ffmpegExtraOptions: str = "-codec libx264 -vf scale=-2:{videoHeight} -pix_fmt yuv420p",
) -> None:
    """Save flythrough as a series of images and then compile into a video"""
    import ScreenCapture

    captureLogic = ScreenCapture.ScreenCaptureLogic()
    if videoDir is None:
        videoDir = imageDir

    # Save a series
    extraOptions = ffmpegExtraOptions.format(videoHeight=videoHeight)
    # Use ScreenCapture module logic to capture a video from the series of images
    videoPathFileName = pathlib.Path.joinpath(videoDir, videoFileName)
    captureLogic.createVideo(
        videoFrameRateFPS, extraOptions, imageDir, imagePattern, videoPathFileName
    )
    # Note that createVideo() automatically specifies -y, -r {frameRate}, -start_number 0
    # If -pix_fmt yuv420p is omitted, many players will not work (incl windows media player)
    # If the scaling is omitted, I get weird artifacts.  One theory is that this is due to bit rate limitations on huge
    # images/videos.
    # Clean up images as requested
    # if deleteImages:
    #    for imgFile in imgFilePaths:
    #        imgFile.unlink()


import time


def test_flight(
    fiducialName, browserName, focIdx, locIdx, mostOpenTimeIdx, flightCurveName
):
    curveNode = getNode(flightCurveName)
    last_loc, last_focus = get_loc_and_foc(getNode(fiducialName), locIdx, focIdx)
    camNode = get_camera_node_for_View1()
    pathStepSpacingMm = 1
    focShift = 5
    camViewUp = (0, -0.5, 1)
    getNode(browserName).SetSelectedItemNumber(mostOpenTimeIdx)
    setup_camera(camNode)
    slicer.modules.markups.logic().JumpSlicesToLocation(
        *last_focus, 0
    )  # 0 is for offset, 1 for centered

    vtkSampledPoints = vtk.vtkPoints()
    slicer.vtkMRMLMarkupsCurveNode.ResamplePoints(
        curveNode.GetCurvePointsWorld(), vtkSampledPoints, pathStepSpacingMm, False
    )
    numPoints = vtkSampledPoints.GetNumberOfPoints()
    for pt_ind in range(numPoints):
        loc = vtkSampledPoints.GetPoint(pt_ind)
        # Switch to chosen camera location as last point
        if pt_ind == numPoints - 1:
            loc = last_loc  # last_loc isn't an input...
        # Decide on focal point (either focShift points ahead, or last_focus)
        if pt_ind + focShift >= numPoints - 1:
            foc = last_focus
        else:
            foc = vtkSampledPoints.GetPoint(pt_ind + focShift)
        # Move the camera
        camNode.SetPosition(loc)
        camNode.SetFocalPoint(foc)
        camNode.SetViewUp(camViewUp)
        slicer.app.processEvents()
        time.sleep(0.1)


def jump_camera(
    location,
    focalPoint,
    cameraNode=None,
    cameraClippingRange=None,
    cameraViewAngleDeg=None,
    cameraViewUpGuideVector=None,
    jumpSlicesMode=-1,
):
    # Default camera node to the camera for View1
    if cameraNode is None:
        # Default to the current camera for the first 3D view ('View1')
        layoutManager = slicer.app.layoutManager()
        for threeDViewIndex in range(layoutManager.threeDViewCount):
            view = layoutManager.threeDWidget(threeDViewIndex).threeDView()
            threeDViewNode = view.mrmlViewNode()
            viewName = threeDViewNode.GetName()
            if viewName == "View1":
                cameraNode = slicer.modules.cameras.logic().GetViewActiveCameraNode(
                    threeDViewNode
                )
    # Set camera view angle if requested
    if cameraViewAngleDeg is not None:
        cameraNode.GetCamera().SetViewAngle(cameraViewAngleDeg)
    # Set camera clipping range if reqested
    if cameraClippingRange is not None:
        cameraNode.GetCamera().SetClippingRange(
            *cameraClippingRange
        )  # format like [0.08, 80]
    # Set focal point and location
    cameraNode.SetFocalPoint(focalPoint)
    cameraNode.SetPosition(location)
    # Orient so projection of guide vector into camera plane is up on camera image
    # NOTE: this will fail badly if guide vector approaches parallel with view direction!
    if cameraViewUpGuideVector is not None:
        cameraNode.SetViewUp(cameraViewUpGuideVector)
    # Handle jumping other slices locations (0 = jump centered, 1 = jump offset, anything else, no jumping)
    if jumpSlicesMode == slicer.vtkMRMLSliceNode.CenteredJumpSlice:  # 0
        slicer.vtkMRMLSliceNode.JumpAllSlices(
            slicer.mrmlScene, *location, slicer.vtkMRMLSliceNode.CenteredJumpSlice
        )
    elif jumpSlicesMode == slicer.vtkMRMLSliceNode.OffsetJumpSlice:  # 1
        slicer.vtkMRMLSliceNode.JumpAllSlices(
            slicer.mrmlScene, *location, slicer.vtkMRMLSliceNode.OffsetJumpSlice
        )
    # Force application update
    slicer.app.processEvents()


def fly_camera(
    locations,
    focalPoints,
    cameraNode=None,
    timeSleepIntervalSec=0.1,
    cameraClippingRange=[0.08, 80],
    cameraViewAngleDeg=110,
    cameraViewUpGuideVector=[0, 0, 1],
    jumpSlicesMode=-1,
):
    """Fly a camera along a series of locations, pointed at a paired series of focal points.
    Current defaults for other parameters are approximately appropriate for virtual endoscopy cameras
    """
    # Validate inputs
    # TODO locations and focal points should have equal shape, and should be numpy arrays with exactly 3 columns
    # TODO what should happen if the viewUp vector is normal to the camera view plane? Do we need a secondary fallback?
    for loc, foc in zip(locations, focalPoints):
        jump_camera(
            loc,
            foc,
            cameraNode=cameraNode,
            cameraClippingRange=cameraClippingRange,
            cameraViewAngleDeg=cameraViewAngleDeg,
            cameraViewUpGuideVector=cameraViewUpGuideVector,
        )
        if jumpSlicesMode == slicer.vtkMRMLSliceNode.CenteredJumpSlice:  # 0
            slicer.vtkMRMLSliceNode.JumpAllSlices(
                slicer.mrmlScene, *loc, slicer.vtkMRMLSliceNode.CenteredJumpSlice
            )
        elif jumpSlicesMode == slicer.vtkMRMLSliceNode.OffsetJumpSlice:  # 1
            slicer.vtkMRMLSliceNode.JumpAllSlices(
                slicer.mrmlScene, *loc, slicer.vtkMRMLSliceNode.OffsetJumpSlice
            )

        time.sleep(timeSleepIntervalSec)


def resample_curve_node(
    curvePointsNode, pathStepSpacingMm=0.5, outputCurveNodeFlag=False
):
    vtkResampledPoints = vtk.vtkPoints()  # initialize
    slicer.vtkMRMLMarkupsCurveNode.ResamplePoints(
        curvePointsNode.GetCurvePointsWorld(),
        vtkResampledPoints,
        pathStepSpacingMm,
        curvePointsNode.GetCurveClosed(),
    )
    resampledControlPoints = vtk.util.numpy_support.vtk_to_numpy(
        vtkResampledPoints.GetData()
    )
    if outputCurveNodeFlag:
        resampledCurveNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsCurveNode",
            slicer.mrmlScene.GenerateUniqueName(
                f"{curvePointsNode.GetName()}_resampled"
            ),
        )
        slicer.util.updateMarkupsControlPointsFromArray(
            resampledCurveNode, resampledControlPoints
        )
        return resampledCurveNode
    else:
        return resampledControlPoints


def createFocalPointsFromLocations(locations, focusDelta=1, finalFocus=None):
    """Create camera focal points from a list of locations and (optionally)
    a final point to look towards.  The camera is always pointed at a location
    which is focusDelta locations ahead of the current location.  Once that is
    no longer possible because the end of the list of locations is being approached,
    the camera is directed towards the finalFocus point (if supplied). If no
    finalFocus point is supplied, a default one is calculated which is just
    past the last location, in the direction of the last location jump.
    """
    focusDelta = int(focusDelta)
    focalPoints = np.zeros(locations.shape)
    focalPoints[:-focusDelta, :] = locations[focusDelta:, :]
    if finalFocus is None:
        # Extrapolate final direction one more step
        lastLoc = locations[-1, :]
        prevLoc = locations[-2, :]
        finalFocus = lastLoc + (lastLoc - prevLoc)
    finalFocus = np.reshape(np.array(finalFocus), (1, 3))  # force to 1x3 numpy array
    focalPoints[-focusDelta:, :] = np.repeat(finalFocus, focusDelta, axis=0)
    return focalPoints


import logging
import pathlib

DEFAULT_VIDEO_PARAMETERS_DICT = {
    "captureVideoFlag": True,
    "tempImageCapturePath": pathlib.Path(slicer.app.temporaryPath).joinpath(
        "tempImgCapFolder"
    ),
    "imageFilePattern": "image_%05d.png",
    "viewToCapture": None,  # None defaults to all views
    "frameRate": 5,  # frames per second
    "videoHeight": 1440,
    "videoFilePath": None,
    "deleteTempImages": True,  # Only make this false during active debugging of problems, otherwise there is potiential for name collisions and reuse of inappropriate images
}


def fly_along_curve(
    curvePointsNode,
    pathStepSpacingMm=0.5,
    focusDelta=1,
    finalFocus=None,
    viewUpGuideVector=[0, 0, 1],
    camNode=None,
    timeSleepIntervalSec=0.1,
    cameraClippingRange=[0.08, 80],
    cameraViewAngleDeg=110,
    jumpSlicesMode=0,
    videoParametersDict=DEFAULT_VIDEO_PARAMETERS_DICT,
):
    """A supplied markupsCurveNode provides the flight path for a camera (default view1). If pathStepSpacingMm
    greater than zero is supplied, the curve described by the markupsCurveNode is resampled to this spacing; if
    pathStepSpacingMm is less than or equal to zero, then the points are not resampled: the sequence if input points
    is the sequence of camera positions.  From each camera location, the camera focal point is the point +focusDelta
    points further along, unless this is past the end of the curve, in which case the focal point is given by the
    finalFocus coordinate (if provided), or defaults to a point projected linearly along the vector connecting the
    second to last point to the last point.   The camera viewUp direction is the projection of the
    viewUpGuideVector onto the plane perpendicular to the camera view direction, and the guide vector defaults to
    superior. By default, the camera node which is moved is the camera for View1 (ther first 3D view), but this can
    be overruled by supplying a camNode input.
    """
    # Default camNode for View1
    if camNode is None:
        camNode = get_camera_node_for_View1()
    # Force focusDelta to positive integer (at least 1)
    focusDelta = max(abs(int(focusDelta)), int(1))
    # Determine resampled points (if needed)
    if pathStepSpacingMm is None or pathStepSpacingMm <= 0:
        # Use the raw points (don't resample)
        cameraLocations = slicer.util.arrayFromMarkupsControlPoints(curvePointsNode)
    else:
        # Resample
        cameraLocations = resample_curve_node(
            curvePointsNode, pathStepSpacingMm=pathStepSpacingMm
        )
    # Get camera focal points
    cameraFocalPoints = createFocalPointsFromLocations(
        cameraLocations, focusDelta=focusDelta, finalFocus=finalFocus
    )

    # Setup for video capture (if needed)
    captureVideoFlag = (
        videoParametersDict["captureVideoFlag"]
        if videoParametersDict is not None
        else False
    )
    if captureVideoFlag:
        import ScreenCapture

        tmpImgCapPath = pathlib.Path(slicer.app.temporaryPath).joinpath(
            "tempImgCapFolder"
        )
        tmpImgCapPath.mkdir(parents=True, exist_ok=True)
        cap = ScreenCapture.ScreenCaptureLogic()
        viewToCapture = videoParametersDict["viewToCapture"]
        imgFilePattern = videoParametersDict["imageFilePattern"]
        frameRate = videoParametersDict["frameRate"]
        videoHeight = videoParametersDict["videoHeight"]
        removeTempImages = videoParametersDict["deleteTempImages"]
        try:
            videoFilePath = pathlib.Path(videoParametersDict["videoFilePath"])
        except KeyError:
            # No dictionary key, default to image path and 'SlicerCapture.mp4'
            videoFilePath = tmpImgCapPath.joinpath("SlicerCapture.mp4")
        except TypeError:
            # Likely dictionary entry was None
            videoFilePath = tmpImgCapPath.joinpath("SlicerCapture.mp4")
        # Ensure parent directory exists
        videoFilePath.parent.mkdir(parents=True, exist_ok=True)

    #### Main Loop Through Locations ####
    for idx, (loc, foc) in enumerate(zip(cameraLocations, cameraFocalPoints)):
        jump_camera(
            loc,
            foc,
            cameraNode=camNode,
            cameraClippingRange=cameraClippingRange,
            cameraViewAngleDeg=cameraViewAngleDeg,
            cameraViewUpGuideVector=viewUpGuideVector,
            jumpSlicesMode=jumpSlicesMode,
        )
        if captureVideoFlag:
            # Capture image to temp
            cap.captureImageFromView(
                viewToCapture, tmpImgCapPath.joinpath(imgFilePattern % idx)
            )
        else:
            time.sleep(timeSleepIntervalSec)
    # Build video from images and clean up temporary folder
    if captureVideoFlag:
        extraOptions = f"-codec libx264 -vf scale=-2:{videoHeight} -pix_fmt yuv420p"
        cap.createVideo(
            frameRate, extraOptions, tmpImgCapPath, imgFilePattern, videoFilePath
        )
        logging.info(f"Created video in {videoFilePath}!")
        if removeTempImages:
            idx = 0
            while pathlib.Path(tmpImgCapPath.joinpath(imgFilePattern % idx)).is_file():
                pathlib.Path(tmpImgCapPath.joinpath(imgFilePattern % idx)).unlink()
                idx += 1

    return


def capture_flight_image(saveDir, filePattern="image_%05d.png", viewToCapture=None):
    import ScreenCapture
    import os

    try:
        os.mkdir(saveDir)
        print("Created directory: %s" % saveDir)
    except FileExistsError:
        pass  # print('Save directory %s already exists'%saveDir)
    cap = ScreenCapture.ScreenCaptureLogic()
    cap.captureImageFromView(viewToCapture, os.path.join(saveDir, f""))


def show_only(F, showIdx):
    for idx in range(F.GetNumberOfControlPoints()):
        if idx == showIdx:
            F.SetNthControlPointVisibility(idx, True)
        else:
            F.SetNthControlPointVisibility(idx, False)


def get_loc_and_foc(F, locIdx, focIdx):
    foc = [0, 0, 0]
    loc = [0, 0, 0]
    F.GetNthControlPointPositionWorld(focIdx, foc)
    F.GetNthControlPointPositionWorld(locIdx, loc)
    return (loc, foc)


def setup_camera(camNode):
    # Set up camera for acquisition
    camNode.SetViewAngle(110)
    camNode.GetCamera().SetClippingRange(0.08, 80)


def set_sequence_segment_color(
    seqBrowserNode, proxySegNode, segmentName, color=(1.0, 0.63, 0.45)
):
    for idx in range(seqBrowserNode.GetNumberOfItems()):
        seqBrowserNode.SetSelectedItemNumber(idx)
        segmentID = proxySegNode.GetSegmentation().GetSegmentIdBySegmentName(
            segmentName
        )
        proxySegNode.GetSegmentation().GetSegment(segmentID).SetColor(color)


def save_video(
    videoFileName,
    imageDirectory,
    imagePattern="image_%05d.png",
    frameRate=5,
    videoHeight=1440,
    deleteImages=False,
):
    # Save a video from the series of captured images.  imagePattern can be as default or glob style *.png (I think)
    import ScreenCapture

    extraOptions = f"-codec libx264 -vf scale=-2:{videoHeight} -pix_fmt yuv420p"
    ScreenCapture.ScreenCaptureLogic().createVideo(
        frameRate, extraOptions, imageDirectory, imagePattern, videoFileName
    )
    # Note that createVideo() automatically specifies -y, -r {frameRate}, -start_number 0
    # If -pix_fmt yuv420p is omitted, many players will not work (incl windows media player)
    # If the scaling is omitted, I get weird artifacts.  One theory is that this is due to bit rate limitations on huge
    # images/videos.  It's not clear how MATLAB's encoder avoids this problem, but it does.
    #
    # The output video is saved to the same directory that the image series is in.
    if deleteImages:
        import re

        allowedFormatSpecPatt = re.compile(
            "\%(0[1-9])?d"
        )  # %d and %04d are allowed formats
        starPatt = re.compile("\*")
        if allowedFormatSpecPatt.search(imagePattern):
            # There is a format specifier, start from 0 and go up until you run out of files
            imgNum = 0
            while os.path.isfile(os.path.join(imageDirectory, imagePattern % imgNum)):
                os.remove(os.path.join(imageDirectory, imagePattern % imgNum))
                imgNum = imgNum + 1
        elif starPatt.search(imagePattern):
            import pathlib

            for img in pathlib.Path(imageDirectory).glob(imagePattern):
                os.path.remove(img)


def setup_lighting():
    # Set some default lighting parameters using the Lights module to make them work OK for
    # the virtual endoscopy view...
    import Lights

    lightLogic = Lights.LightsLogic()
    view1node = slicer.mrmlScene.GetNodeByID("vtkMRMLViewNode1")  # view1, first 3D view
    lightLogic.addManagedView(view1node)
    lightKit = lightLogic.lightKit
    #
    lightKit.SetKeyLightIntensity(0.8)
    # Make key light act like headlight (point straight ahead from camera)
    lightKit.SetKeyLightElevation(0)
    lightKit.SetKeyLightAzimuth(0)
    # Brighten head light
    lightKit.SetKeyToHeadRatio(1)
    # Fill light keeps the upper part from being too dark
    lightKit.SetKeyToFillRatio(1.5)
    # The back light default seems like it works well (KeyToBackRatio=3.5)
    # Turn on ambient shadows (looks a bit better)
    lightLogic.setUseSSAO(1)
    lightLogic.setSSAOSizeScaleLog(0)


def run_4D(
    fiducialName,
    videoName,
    browserName,
    fps,
    focIdx,
    locIdx,
    mostOpenTimeIdx,
    flightCurveName,
    saveDir,
    numFullDynamicLoops=2,
    moveSlices=False,
):
    # Gather nodes
    camNode = get_camera_node_for_View1()
    cam2 = get_camera_node_for_View2()
    F = getNode(fiducialName)
    browserNode = getNode(browserName)
    curveNode = getNode(flightCurveName)

    show_only(F, focIdx)
    (loc, foc) = get_loc_and_foc(F, locIdx, focIdx)
    slicer.modules.markups.logic().JumpSlicesToLocation(
        *foc, 0
    )  # 0 is for offset, 1 for centered
    setup_camera(camNode)
    setup_lighting()
    coneHeight = get_cone_height(loc, foc)

    last_loc = loc
    last_focus = foc
    browserNode.SetSelectedItemNumber(mostOpenTimeIdx)
    imgFilePattern = "image_%05d.png"
    make_flight_capture(
        curveNode,
        last_loc,
        last_focus,
        camNode,
        coneHeight,
        browserNode,
        mostOpenTimeIdx,
        saveDir,
        filePattern=imgFilePattern,
        pathStepSpacingMm=1,
        coneDynamicOpacity=0.5,
        numFullDynamicLoops=numFullDynamicLoops,
        moveSlices=moveSlices,
    )
    # Print camera parameters so it is possible to reproduce 3D camera positioning
    print_cam_params(cam2)
    # Save the video (new MATLAB-free method)
    save_video(
        videoName,
        saveDir,
        imagePattern=imgFilePattern,
        frameRate=fps,
        videoHeight=1440,
        deleteImages=False,
    )
    # could change deleteImages default to True once there's more confidence in this procedure
    ### Print commmand for MATLAB video assembly
    # print("make_4D_video('%s','%s',%i)"%(saveDir,videoName,fps))
