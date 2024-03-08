import logging
import os
from typing import Annotated, Optional
import pathlib
import numpy as np
from numpy.typing import ArrayLike
import enum
import time

import vtk
import qt

import slicer

if (slicer.app.majorVersion, slicer.app.minorVersion) > (5, 3):
    # No i18n in 5.2.1
    from slicer.i18n import tr as _
    from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import parameterNodeWrapper, WithinRange, Choice

from slicer import (
    vtkMRMLScalarVolumeNode,
    vtkMRMLMarkupsNode,
    vtkMRMLMarkupsCurveNode,
    vtkMRMLCameraNode,
    vtkMRMLSliceNode,
    vtkMRMLViewNode,
    vtkMRMLSequenceBrowserNode,
)

import ScreenCapture
from Resources.FromJupyter import jupyterNbFcns


#
class JumpSliceModeEnum(enum.Enum):
    CENTERED = vtkMRMLSliceNode.CenteredJumpSlice
    OFFSET = vtkMRMLSliceNode.OffsetJumpSlice
    NONE = -1

    def label(self):
        """This will be used to define the strings appearing in a dropdown choice
        corresponding to each option
        """
        if self.name == "CENTERED":
            choiceString = "Jump slice views (centered)"
        elif self.name == "OFFSET":
            choiceString = "Jump slice views (offset)"
        else:
            choiceString = "Don't jump slice views"
        return choiceString


class VideoRecordingModeEnum(enum.Enum):
    ALL_VIEWS = 0
    THREE_D_ONLY = 1

    def label(self):
        if self.name == "ALL_VIEWS":
            choiceString = "record 3D view + slice views"
        elif self.name == "THREE_D_ONLY":
            choiceString = "record only 3D view"
        else:
            raise (Exception(f"Unexpected VideoRecordingModeEnum name {self.name}!"))
        return choiceString


#
# VirtualEndoscopy
#


class VirtualEndoscopy(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _(
            "VirtualEndoscopy"
        )  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = (
            []
        )  # TODO: add here list of module names that this module requires
        self.parent.contributors = [
            "John Doe (AnyWare Corp.)"
        ]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _(
            """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#VirtualEndoscopy">module documentation</a>.
"""
        )
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _(
            """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""
        )

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # VirtualEndoscopy1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="VirtualEndoscopy",
        sampleName="VirtualEndoscopy1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "VirtualEndoscopy1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="VirtualEndoscopy1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="VirtualEndoscopy1",
    )

    # VirtualEndoscopy2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="VirtualEndoscopy",
        sampleName="VirtualEndoscopy2",
        thumbnailFileName=os.path.join(iconsPath, "VirtualEndoscopy2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="VirtualEndoscopy2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="VirtualEndoscopy2",
    )


#
# VirtualEndoscopyParameterNode
#


@parameterNodeWrapper
class VirtualEndoscopyParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputCurve: vtkMRMLMarkupsCurveNode
    resampleSpacingMm: float = 0.5
    useResampleSpacingBool: bool = True
    lookAheadIntervalPoints: int = 5
    cameraLocationsCurveNode: vtkMRMLMarkupsCurveNode
    cameraNode: vtkMRMLCameraNode
    focalPointsCurveNode: vtkMRMLMarkupsCurveNode
    finalFocalPoint: vtkMRMLMarkupsNode  # could be either curve node or points list node (or... only the first control point will be used)
    viewUpGuideVectorR: float = 0
    viewUpGuideVectorA: float = 0
    viewUpGuideVectorS: float = 1
    useViewUpGuideVectorBool: bool = True
    cameraViewAngleDegrees: Annotated[float, WithinRange(1, 360)] = 110
    useCameraViewAngleBool: bool = True
    cameraClippingRangeMinimum: float = 0.08
    cameraClippingRangeMaximum: float = (
        80  # the maximum shouldn't be more than (AT MOST!) about 10K * the minimum, and 1K-3K seems better for most cases
    )
    useCameraClippingRangeBool: bool = True
    playbackTimerIntervalMilliseconds: int = 10
    videoFrameRateFPS: float = 5
    videoHeightPixels: int = 1440
    videoRecordingMode: VideoRecordingModeEnum
    videoSaveFilePath: pathlib.Path = pathlib.Path.home().joinpath(
        "SlicerCapture", "VirtualEndoVideo.mp4"
    )
    currentStepIndex: Annotated[float, WithinRange(0, 200000)] = (
        0  # should be int, but needs to be float to connect to slider
    )
    currentlyPlaying: bool = False
    currentlyRecordingVideo: bool = False
    jumpSliceViewMode: JumpSliceModeEnum
    numberOfSteps: int = 0


#
# VirtualEndoscopyWidget
#


class VirtualEndoscopyWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self.timer = qt.QTimer()
        self.timer.setInterval(20)
        self.timer.connect("timeout()", self.jumpToNext)

    def setup(self) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/VirtualEndoscopy.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = VirtualEndoscopyLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose
        )
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose
        )

        # Buttons
        self.ui.playPushButton.connect("toggled(bool)", self.onPlayPushButtonToggled)
        self.ui.recordPushButton.connect(
            "clicked(bool)", self.onRecordPushButtonClicked
        )
        self.ui.preprocessPushButton.connect(
            "clicked(bool)", self.onPreprocessButtonClicked
        )
        # Slider
        self.ui.stepSliderWidget.connect(
            "valueChanged(double)", self.onStepSliderValueChanged
        )

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """
        Called when the application closes and the module widget is destroyed.
        """
        # I seem to need to disconnect any connections here also in the
        # case of module reloading (irrelevant but harmless in the case
        # of closing Slicer or anything else which unloads the module)
        self.ui.stepSliderWidget.disconnect(
            "valueChanged(double)", self.onStepSliderValueChanged
        )
        self.ui.playPushButton.disconnect("toggled(bool)", self.onPlayPushButtonToggled)
        self.ui.recordPushButton.disconnect(
            "clicked(bool)", self.onRecordPushButtonClicked
        )
        self.ui.preprocessPushButton.disconnect(
            "clicked(bool)", self.onPreprocessButtonClicked
        )
        # Remove all observers of the module widget
        self.removeObservers()

    def enter(self) -> None:
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self._onParameterNodeModified,
            )

    def onSceneStartClose(self, caller, event) -> None:
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Template showed selection of input node if nothing is selected yet here,
        # but we don't need that functionality.  Other initialization code
        # could go here if needed

    def setParameterNode(
        self, inputParameterNode: Optional[VirtualEndoscopyParameterNode]
    ) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self._onParameterNodeModified,
            )
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self._onParameterNodeModified,
            )
            self._onParameterNodeModified()

    def _onParameterNodeModified(self, caller=None, event=None) -> None:
        """
        Triggered whenever the parameter node is modified (GUI or code!). Should
        manage whether buttons are enabled or not.  Could also be set up to detect
        whether the processed curves are up to date if we cache the previous parameter
        node state.
        """
        pn = self._parameterNode
        # Preprocess button (enabled if an input curve node is selected)
        self.ui.preprocessPushButton.enabled = True if pn.inputCurve else False
        # Enable Play button And Record Button if both locations and focal points curves are selected
        enablePlayBool = (
            pn.cameraLocationsCurveNode
            and pn.focalPointsCurveNode
            and pn.cameraLocationsCurveNode.GetNumberOfControlPoints() >= 1
        )
        self.ui.playPushButton.enabled = enablePlayBool
        enableRecordBool = enablePlayBool and pn.videoSaveFilePath
        self.ui.recordPushButton.enabled = enableRecordBool

        # Set Maximum Steps to length of locations curve (-1)
        if (
            pn.cameraLocationsCurveNode is None
            or pn.cameraLocationsCurveNode.GetNumberOfControlPoints() < 1
        ):
            pn.numberOfSteps = 0
            self.ui.stepSliderWidget.maximum = (
                200  # default to 200 if no curve selected
            )
        else:
            pn.numberOfSteps = pn.cameraLocationsCurveNode.GetNumberOfControlPoints()
            self.ui.stepSliderWidget.maximum = pn.numberOfSteps - 1

        #
        self.timer.setInterval(pn.playbackTimerIntervalMilliseconds)
        print("modified")

    def jumpToNext(self):
        """Jump to the next step."""
        pn = self._parameterNode
        if pn.currentStepIndex >= pn.numberOfSteps - 1:
            self.stopPlaying()  # or could loop back to step 0...?
        else:
            pn.currentStepIndex = pn.currentStepIndex + 1

    def onStepSliderValueChanged(self, stepValue) -> None:
        """Called whenever step slider value changes"""
        pn = self._parameterNode
        location, focalPoint = self.logic.getNthLocAndFoc(
            pn.cameraLocationsCurveNode, pn.focalPointsCurveNode, int(stepValue)
        )
        # gather camera clipping range from components
        cameraClippingRange = (
            [pn.cameraClippingRangeMinimum, pn.cameraClippingRangeMaximum]
            if pn.useCameraClippingRangeBool
            else None
        )
        # Gather view up guide vector from components
        viewUpGuideVector = (
            [pn.viewUpGuideVectorR, pn.viewUpGuideVectorA, pn.viewUpGuideVectorS]
            if pn.useViewUpGuideVectorBool
            else None
        )

        self.logic.jumpCamera(
            location,
            focalPoint,
            cameraNode=pn.cameraNode,
            cameraClippingRange=cameraClippingRange,
            cameraViewAngleDeg=pn.cameraViewAngleDegrees,
            cameraViewUpGuideVector=viewUpGuideVector,
            jumpSlicesMode=pn.jumpSliceViewMode,
        )
        print(f"New step is {stepValue}")

    def onPlayPushButtonToggled(self, toggleBool) -> None:
        """Starting from the current step, play through all remaining steps"""
        if toggleBool:
            self.startPlaying()
        else:
            self.stopPlaying()

    def startPlaying(self):
        self.timer.start()
        self.ui.playPushButton.text = "Stop"

    def stopPlaying(self):
        self.timer.stop()
        self.ui.playPushButton.text = "Play"

    def onRecordPushButtonClicked(self) -> None:
        """Record video of full flythrough"""
        # Gather parameters
        # Capture the video
        self.logic.recordVideo(self._parameterNode)
        pass

    def onPreprocessButtonClicked(self) -> None:
        """
        Preprocess input curve to camera locations and focal points
        """
        # Gather inputs
        inputCurveNode = self._parameterNode.inputCurve
        useResampleSpacing = self._parameterNode.useResampleSpacingBool
        resampleSpacingMm = self._parameterNode.resampleSpacingMm
        lookAheadInterval = self._parameterNode.lookAheadIntervalPoints
        # Gather or create output nodes
        if not self._parameterNode.cameraLocationsCurveNode:
            camLocCurveName = slicer.mrmlScene.GenerateUniqueName(
                inputCurveNode.GetName() + "_CamLocs"
            )
            self._parameterNode.cameraLocationsCurveNode = (
                slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLMarkupsCurveNode", camLocCurveName
                )
            )
        cameraLocations = self._parameterNode.cameraLocationsCurveNode
        if not self._parameterNode.focalPointsCurveNode:
            camFocCurveName = slicer.mrmlScene.GenerateUniqueName(
                inputCurveNode.GetName() + "_CamFoci"
            )
            self._parameterNode.focalPointsCurveNode = (
                slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLMarkupsCurveNode", camFocCurveName
                )
            )
        cameraFocalPoints = self._parameterNode.focalPointsCurveNode
        # Process
        self.logic.processInputCurveToLocationsAndFocalPoints(
            inputCurveNode,
            useResampleSpacing,
            resampleSpacingMm,
            lookAheadInterval,
            cameraLocations,
            cameraFocalPoints,
        )
        # Hide curves so that they don't interfere with endoscopy
        self.logic.hideCurve(cameraFocalPoints)
        self.logic.hideCurve(cameraLocations)
        # Probably un-needed!
        # self._onParameterNodeModified()

    def onApplyButton(self) -> None:
        """
        Run processing when user clicks "Apply" button.
        """
        with slicer.util.tryWithErrorDisplay(
            _("Failed to compute results."), waitCursor=True
        ):
            # Compute output
            self.logic.process(
                self.ui.inputSelector.currentNode(),
                self.ui.outputSelector.currentNode(),
                self.ui.imageThresholdSliderWidget.value,
                self.ui.invertOutputCheckBox.checked,
            )

            # Compute inverted output (if needed)
            if self.ui.invertedOutputSelector.currentNode():
                # If additional output volume is selected then result with inverted threshold is written there
                self.logic.process(
                    self.ui.inputSelector.currentNode(),
                    self.ui.invertedOutputSelector.currentNode(),
                    self.ui.imageThresholdSliderWidget.value,
                    not self.ui.invertOutputCheckBox.checked,
                    showResult=False,
                )


#
# VirtualEndoscopyLogic
#


class VirtualEndoscopyLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return VirtualEndoscopyParameterNode(super().getParameterNode())

    def hideCurve(self, curveNode: vtkMRMLMarkupsCurveNode) -> None:
        """Set curve visibility to false"""
        curveNode.GetDisplayNode().SetVisibility(0)

    def getThreeDViewNodeByName(self, name="View1") -> vtkMRMLViewNode:
        """Find the vtkMRMLViewNode with the given name.  A ViewNameNotFoundError is
        returned if no 3D views with a matching name are found.  If no name is supplied,
        a default value of "View1" is used."""
        lm = slicer.app.layoutManager()
        for viewIdx in range(lm.threeDViewCount):
            viewNode = lm.threeDWidget(viewIdx).mrmlViewNode()
            if viewNode.GetName() == name:
                return viewNode
        # Name failed to match any 3D view
        raise ViewNameNotFoundError(f'View named "{name}" not found!')

    def recordVideo(
        self,
        parameterNode: VirtualEndoscopyParameterNode,
        deleteImages: bool = True,
        ffmpegExtraOptions: str = "-codec libx264 -vf scale=-2:{videoHeight} -pix_fmt yuv420p",
    ):
        """Save flythrough as a series of images and then compile into a video"""
        captureLogic = ScreenCapture.ScreenCaptureLogic()

        imgFilePaths, imagePattern = self.saveImageSeriesByParameterNode(
            parameterNode=parameterNode
        )
        frameRate = parameterNode.videoFrameRateFPS
        videoHeight = parameterNode.videoHeightPixels

        # Save a series
        extraOptions = ffmpegExtraOptions.format(videoHeight=videoHeight)
        # Use ScreenCapture module logic to capture a video from the series of images
        videoFileName = parameterNode.videoSaveFilePath
        imageDirectory = imgFilePaths[0].parent
        captureLogic.createVideo(
            frameRate, extraOptions, imageDirectory, imagePattern, videoFileName
        )
        # Note that createVideo() automatically specifies -y, -r {frameRate}, -start_number 0
        # If -pix_fmt yuv420p is omitted, many players will not work (incl windows media player)
        # If the scaling is omitted, I get weird artifacts.  One theory is that this is due to bit rate limitations on huge
        # images/videos.
        # Clean up images as requested
        if deleteImages:
            for imgFile in imgFilePaths:
                imgFile.unlink()

    def saveImageSeriesByParameterNode(
        self,
        parameterNode: VirtualEndoscopyParameterNode,
        imageDirectory: Optional[pathlib.Path] = None,
        imageFilePattern: str = "tempImage_%05d.png",
    ) -> tuple[list[pathlib.Path], str]:
        """Save image series using parameters from parameter node.
        Return list of saved image file names and the image file pattern used
        """
        captureLogic = ScreenCapture.ScreenCaptureLogic()
        # Determine view to capture (3D only or all views)
        if parameterNode.videoRecordingMode == VideoRecordingModeEnum.THREE_D_ONLY:
            viewToCapture = self.getThreeDViewNodeByName()
        elif parameterNode.videoRecordingMode == VideoRecordingModeEnum.ALL_VIEWS:
            viewToCapture = None  # ScreenCapture treats this as 'capture all views'
        else:
            raise ValueError(
                f"Unknown video recording mode {parameterNode.videoRecordingMode} supplied!"
            )
        # Determine Image Directory
        if imageDirectory is None:
            # Create a temporary subdirectory of video directory
            vidSaveFilePath = parameterNode.videoSaveFilePath
            vidSaveDirectory = vidSaveFilePath.parent
            imageDirectory = pathlib.Path.joinpath(vidSaveDirectory, "TempImageDir")
        # Ensure image directory exists for saving
        imageDirectory.mkdir(parents=True, exist_ok=True)

        imageFilePaths = []

        for stepNumber in range(parameterNode.numberOfSteps):
            parameterNode.currentStepIndex = stepNumber
            self.jumpCameraByParameterNode(parameterNode=parameterNode)
            imageFileName = imageFilePattern % (stepNumber)
            imageFilePath = pathlib.Path(imageDirectory, imageFileName)
            captureLogic.captureImageFromView(
                view=viewToCapture, filename=imageFilePath
            )
            imageFilePaths.append(imageFilePath)
        return imageFilePaths, imageFilePattern

    def getNthLocAndFoc(
        self,
        locationsCurveNode: vtkMRMLMarkupsNode,
        focalPointsCurveNode: vtkMRMLMarkupsNode,
        stepToRetrieve: int,
    ) -> tuple[ArrayLike, ArrayLike]:
        """Return the Nth control point world position from two curve nodes representing
        camera locations and camera focal points."""
        location = locationsCurveNode.GetNthControlPointPositionWorld(stepToRetrieve)
        focalPoint = focalPointsCurveNode.GetNthControlPointPositionWorld(
            stepToRetrieve
        )
        return location, focalPoint

    def processInputCurveToLocationsAndFocalPoints(
        self,
        inputCurveNode: vtkMRMLMarkupsNode,
        useResampleSpacing: bool,
        resampleSpacingMm: float,
        lookAheadInterval: int,
        cameraLocations: Optional[vtkMRMLMarkupsNode] = None,
        cameraFocalPoints: Optional[vtkMRMLMarkupsNode] = None,
        finalFocalPoint: Optional[list[float]] = None,
    ) -> tuple[vtkMRMLMarkupsNode, vtkMRMLMarkupsNode]:
        """
        Given an input curve, this function typically resamples it to a uniformly spaced set of camera
        locations following the curve, as well as a set of camera focal points which are just the
        """
        if cameraLocations is None:
            cameraLocationsName = slicer.mrmlScene.GenerateUniqueName(
                f"{inputCurveNode.GetName()}_CamLocs"
            )
            cameraLocations = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsCurveNode", cameraLocationsName
            )
        if cameraFocalPoints is None:
            cameraFocalPointsName = slicer.mrmlScene.GenerateUniqueName(
                f"{inputCurveNode.GetName()}_CamFoci"
            )
            cameraFocalPoints = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsCurveNode", cameraFocalPointsName
            )
        # Resample if requested
        if useResampleSpacing:
            self.resampleCurveNode(inputCurveNode, resampleSpacingMm, cameraLocations)
        # Derive camera focal point locations
        self.createOrUpdateFocalPointNode(
            cameraLocations, lookAheadInterval, finalFocalPoint, cameraFocalPoints
        )

        return cameraLocations, cameraFocalPoints

    def createOrUpdateFocalPointNode(
        self,
        cameraLocations: vtkMRMLMarkupsNode,
        lookAheadInterval: int,
        finalFocalPoint: ArrayLike,
        cameraFocalPoints: Optional[vtkMRMLMarkupsNode] = None,
    ) -> vtkMRMLMarkupsNode:
        """
        Generates camera focal points which are lookAheadInterval steps ahead of the
        current camera location. When the end of the list of the locations is reached,
        a final focal point is used, which is finalFocalPoint if supplied, or a
        forward projection of the last step if not.
        """
        locationsArray = slicer.util.arrayFromMarkupsControlPoints(
            cameraLocations, world=True
        )
        focalPointsArray = self.createFocalPointsArrayFromLocationsArray(
            locationsArray, focusDelta=lookAheadInterval, finalFocus=finalFocalPoint
        )
        if cameraFocalPoints is None:
            # Need to create focal points curve node
            cameraFocalPointsName = slicer.mrmlScene.GenerateUniqueName(
                f"{cameraLocations.GetName()}_CamFoci"
            )
            # TODO: could add parsing to drop "_CamLocs" before adding "_CamFoci" for shorter default names
            cameraFocalPoints = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsCurveNode", cameraFocalPointsName
            )
        slicer.util.updateMarkupsControlPointsFromArray(
            cameraFocalPoints, focalPointsArray, world=True
        )
        return cameraFocalPoints

    def createFocalPointsArrayFromLocationsArray(
        self, locationsArray: np.ndarray, focusDelta: int = 1, finalFocus=None
    ) -> np.ndarray:
        """Create camera focal points from a list of locations and (optionally)
        a final point to look towards.  The camera is always pointed at a location
        which is focusDelta locations ahead of the current location.  Once that is
        no longer possible because the end of the list of locations is being approached,
        the camera is directed towards the finalFocus point (if supplied). If no
        finalFocus point is supplied, a default one is calculated which is just
        past the last location, in the direction of the last location jump.
        """
        # Make sure locationsArray input has the proper orientation
        assert (
            locationsArray.shape[1] == 3
        ), "locationsArray MUST be supplied as an Nx3 numpy array"
        # Initialize and fill the easy part of the focal points array
        # (up until we hit the end of the locations list)
        focalPoints = np.zeros(locationsArray.shape)
        focalPoints[:-focusDelta, :] = locationsArray[focusDelta:, :]
        # Deterimin final focal point if one is not supplied
        if finalFocus is None:
            # Extrapolate final direction one more step
            lastLoc = locationsArray[-1, :]
            prevLoc = locationsArray[-2, :]
            finalFocus = lastLoc + (lastLoc - prevLoc)
        # Force final focal point to correct 1x3 shape (prior code generates
        # an array with shape (3,) rather than (1,3) )
        finalFocus = np.reshape(
            np.array(finalFocus), (1, 3)
        )  # force to 1x3 numpy array
        # Fill the remainder of the focal point locations with this final point
        focalPoints[-focusDelta:, :] = np.repeat(finalFocus, focusDelta, axis=0)
        return focalPoints

    def resampleCurveNode(
        self,
        curvePointsNode: vtkMRMLMarkupsNode,
        pathStepSpacingMm: float = 0.5,
        resampledOutputCurveNode: Optional[vtkMRMLMarkupsNode] = None,
    ) -> vtkMRMLMarkupsCurveNode:
        """
        Uniformly resamples the input curve's control points
        """
        vtkResampledPoints = vtk.vtkPoints()  # initialize
        vtkMRMLMarkupsCurveNode.ResamplePoints(
            curvePointsNode.GetCurvePointsWorld(),
            vtkResampledPoints,
            pathStepSpacingMm,
            curvePointsNode.GetCurveClosed(),
        )
        resampledControlPoints = vtk.util.numpy_support.vtk_to_numpy(
            vtkResampledPoints.GetData()
        )
        # Create output curve node if not supplied
        if resampledOutputCurveNode is None:
            resampledOutputCurveNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsCurveNode",
                slicer.mrmlScene.GenerateUniqueName(
                    f"{curvePointsNode.GetName()}_resampled"
                ),
            )
        # Update output with resampled curve points
        slicer.util.updateMarkupsControlPointsFromArray(
            resampledOutputCurveNode, resampledControlPoints
        )
        return resampledOutputCurveNode

    def jumpCameraByParameterNode(
        self, parameterNode: VirtualEndoscopyParameterNode
    ) -> None:
        """Call jumpCamera, but with parameters supplied by parameter node rather than explicitly"""
        (
            location,
            focalPoint,
            cameraNode,
            cameraClippingRange,
            cameraViewAngleDeg,
            cameraViewUpGuideVector,
            jumpSlicesMode,
        ) = self.gatherJumpCameraParameters(parameterNode)
        self.jumpCamera(
            location,
            focalPoint,
            cameraNode=cameraNode,
            cameraClippingRange=cameraClippingRange,
            cameraViewAngleDeg=cameraViewAngleDeg,
            cameraViewUpGuideVector=cameraViewUpGuideVector,
            jumpSlicesMode=jumpSlicesMode,
        )

    def gatherJumpCameraParameters(
        self, parameterNode: VirtualEndoscopyParameterNode
    ) -> tuple[
        ArrayLike,
        ArrayLike,
        Optional[vtkMRMLCameraNode],
        Optional[ArrayLike],
        Optional[float],
        Optional[ArrayLike],
        Optional[JumpSliceModeEnum],
    ]:
        pn = parameterNode
        location, focalPoint = self.getNthLocAndFoc(
            pn.cameraLocationsCurveNode,
            pn.focalPointsCurveNode,
            int(pn.currentStepIndex),
        )
        # gather camera clipping range from components
        cameraClippingRange = (
            [pn.cameraClippingRangeMinimum, pn.cameraClippingRangeMaximum]
            if pn.useCameraClippingRangeBool
            else None
        )
        # Gather view up guide vector from components
        viewUpGuideVector = (
            [pn.viewUpGuideVectorR, pn.viewUpGuideVectorA, pn.viewUpGuideVectorS]
            if pn.useViewUpGuideVectorBool
            else None
        )
        return (
            location,
            focalPoint,
            pn.cameraNode,
            cameraClippingRange,
            pn.cameraViewAngleDegrees,
            viewUpGuideVector,
            pn.jumpSliceViewMode,
        )

    def jumpCamera(
        self,
        location: ArrayLike,
        focalPoint: ArrayLike,
        cameraNode: Optional[vtkMRMLCameraNode] = None,
        cameraClippingRange: Optional[ArrayLike] = None,
        cameraViewAngleDeg: Optional[float] = None,
        cameraViewUpGuideVector: Optional[ArrayLike] = None,
        jumpSlicesMode: JumpSliceModeEnum = JumpSliceModeEnum.NONE,
    ):
        """
        Jump camera to given location, looking in the direction of the given
        focal point.

        A number of optional parameters allow additional control:
        cameraNode:     camera to control (defaults to the current
                        camera for View1 if None)
        cameraClippingRange:    2 element vector to control camera's near
                                and far clipping planes
        cameraViewAngleDeg:     camera view angle in degrees (unchanged if None)
        cameraViewUpGuideVector:    "Up" in camera image is given by projection
                                    of this vector, prior if None
        jumpSlicesMode:     whether slice intersections should be jumped and
                            if so how (default is yes, and centered)

        """
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
        # Jump slice intersection locations if requested
        if (
            jumpSlicesMode == JumpSliceModeEnum.CENTERED
            or jumpSlicesMode == JumpSliceModeEnum.OFFSET
        ):
            slicer.vtkMRMLSliceNode.JumpAllSlices(
                slicer.mrmlScene, *location, jumpSlicesMode.value
            )
        # Force application update
        slicer.app.processEvents()

    def setupColorSceneLayoutAndLighting(
        self,
        sequenceBrowserNode,
        segmentationProxyNode,
        segmentName="SimpleAirwayAir",
        segmentColor=None,
    ):
        if segmentColor:
            jupyterNbFcns.set_sequence_segment_color(
                sequenceBrowserNode,
                segmentationProxyNode,
                segmentName,
                color=segmentColor,
            )
        else:
            jupyterNbFcns.set_sequence_segment_color(
                sequenceBrowserNode, segmentationProxyNode, segmentName
            )
        # Change layout to Dual3D view if it's not already
        Dual3DLayoutID = 15
        slicer.app.layoutManager().setLayout(Dual3DLayoutID)
        # Set better lighting for endoscopy view
        jupyterNbFcns.setup_lighting()

    """ IDEA for TODO: Instead of doing a full flythrough of the VirtualEndoscopy
    path, and THEN doing dynamic looping, some additional flexibility could be cool.
    Allow specification of a list of frame numbers to stop at and run the 
    dynamic loop(s). This could be helpful if there are possibly multiple
    points of interest. Also, if we could specify the last flythrough index
    we could skip the need to chop off the continuation of the centerline.
    This shouldn't be too hard to implement. 
    """

    def createDynamicVirtualEndoVid(
        self,
        parameterNode: VirtualEndoscopyParameterNode,
        sequenceBrowserNode: vtkMRMLSequenceBrowserNode,
        mostOpenFrameNumber: int,
        numFullDynamicLoops: int = 2,
        ffmpegExtraOptions: str = "-codec libx264 -vf scale=-2:{videoHeight} -pix_fmt yuv420p",
        deleteImages: bool = True,
    ) -> None:
        """This function is to generate a virtual endoscopy video
        for a dynamic CT.  This capability is not yet built in to
        the GUI part of the module because it does not know about
        the sequence browser.
        """
        # Go to most open frame for fly-in
        sequenceBrowserNode.SetSelectedItemNumber(mostOpenFrameNumber)
        # Capture fly-in image series
        imgFilePaths, imgPattern = self.saveImageSeriesByParameterNode(
            parameterNode,
            imageDirectory=None,
        )
        imgSaveDir = imgFilePaths[0].parent
        # Add the dynamic cycling images
        cycleImgFilePaths = jupyterNbFcns.capture_cycles_images(
            sequenceBrowserNode,
            imgSaveDir,
            imgPattern,
            numFullDynamicLoops=numFullDynamicLoops,
            frameStartingIdx=len(imgFilePaths),
        )
        # Assemble into video file!
        captureLogic = ScreenCapture.ScreenCaptureLogic()
        frameRate = parameterNode.videoFrameRateFPS
        videoHeight = parameterNode.videoHeightPixels

        # Save a series
        extraOptions = ffmpegExtraOptions.format(videoHeight=videoHeight)
        # Use ScreenCapture module logic to capture a video from the series of images
        videoFileName = parameterNode.videoSaveFilePath
        captureLogic.createVideo(
            frameRate, extraOptions, imgSaveDir, imgPattern, videoFileName
        )
        # Note that createVideo() automatically specifies -y, -r {frameRate}, -start_number 0
        # If -pix_fmt yuv420p is omitted, many players will not work (incl windows media player)
        # If the scaling is omitted, I get weird artifacts.  One theory is that this is due to bit rate limitations on huge
        # images/videos.
        # Clean up images as requested
        if deleteImages:
            for imgFile in imgFilePaths:
                imgFile.unlink()
            for imgFile in cycleImgFilePaths:
                imgFile.unlink()
            # Also remove the temporary directory
            imgSaveDir.unlink()

    def process(
        self,
        inputVolume: vtkMRMLScalarVolumeNode,
        outputVolume: vtkMRMLScalarVolumeNode,
        imageThreshold: float,
        invert: bool = False,
        showResult: bool = True,
    ) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time

        startTime = time.time()
        logging.info("Processing started")

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            "InputVolume": inputVolume.GetID(),
            "OutputVolume": outputVolume.GetID(),
            "ThresholdValue": imageThreshold,
            "ThresholdType": "Above" if invert else "Below",
        }
        cliNode = slicer.cli.run(
            slicer.modules.thresholdscalarvolume,
            None,
            cliParams,
            wait_for_completion=True,
            update_display=showResult,
        )
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")


#
# VirtualEndoscopyTest
#


class VirtualEndoscopyTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_VirtualEndoscopy1()

    def test_VirtualEndoscopy1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("VirtualEndoscopy1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = VirtualEndoscopyLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")


class ViewNameNotFoundError(Exception):
    pass
