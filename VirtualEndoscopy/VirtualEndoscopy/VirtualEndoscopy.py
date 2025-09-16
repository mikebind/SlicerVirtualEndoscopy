import logging
import os
from typing import Annotated, Optional, List
import pathlib
import numpy as np
from numpy.typing import ArrayLike
import enum
import time
import re

import vtk
import qt

import slicer

if (slicer.app.majorVersion, slicer.app.minorVersion) > (5, 3):
    # No i18n in 5.2.1
    from slicer.i18n import tr as _
    from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin, warningDisplay
from slicer.util import (
    arrayFromMarkupsControlPoints,
    updateMarkupsControlPointsFromArray,
)
from slicer.parameterNodeWrapper import parameterNodeWrapper, WithinRange, Choice

from slicer import (
    vtkMRMLScalarVolumeNode,
    vtkMRMLMarkupsNode,
    vtkMRMLMarkupsCurveNode,
    vtkMRMLCameraNode,
    vtkMRMLSliceNode,
    vtkMRMLViewNode,
    vtkMRMLSequenceBrowserNode,
    qMRMLThreeDView,
    vtkMRMLModelNode,
    vtkMRMLLinearTransformNode,
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


class LayoutSelectionEnum(enum.Enum):
    UNCHANGED = 0
    DUAL_3D = 15
    THREE_D_ONLY = 4

    def label(self):
        if self.name == "UNCHANGED":
            choiceString = "leave unchanged"
        elif self.name == "DUAL_3D":
            choiceString = "dual 3D + slices"
        elif self.name == "THREE_D_ONLY":
            choiceString = "3D only"
        else:
            raise (Exception(f"Unexpected LayoutSelectionEnum name {self.name}!"))
        return choiceString


class LightingModeEnum(enum.Enum):
    UNCHANGED = 0
    POINT_HEADLIGHT = 1
    DIRECTION_HEADLIGHT = 2

    def label(self):
        if self.name == "UNCHANGED":
            choiceString = "leave unchanged"
        elif self.name == "POINT_HEADLIGHT":
            choiceString = "Point source (recommended)"
        elif self.name == "DIRECTION_HEADLIGHT":
            choiceString = "Direction source"
        else:
            raise (Exception(f"Unexpected LightingModeEnum name {self.name}!"))
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
    useSmoothingBool: bool = True
    smoothingWindow: int = 11
    smoothingOrder: int = 1
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
    deleteImages: bool = True
    lightingMode: LightingModeEnum
    useForceLayoutSelection: bool = True
    layoutSelection: LayoutSelectionEnum
    coneModel: vtkMRMLModelNode
    coneTransform: vtkMRMLLinearTransformNode
    # 4D
    sequenceBrowser: vtkMRMLSequenceBrowserNode
    browserFrameForFlythrough: Annotated[float, WithinRange(0, 1000)]
    addCameraConeFlag: bool = True
    stopAfterLastCycleFlag: bool = True
    cycleStepsString: str
    repetitionCount: int = 2
    useStandardize4DSegmentColor: bool = True


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
        self.ui.record4DPushButton.connect(
            "clicked(bool)", self.onRecord4DButtonClicked
        )
        # Slider
        self.ui.stepSliderWidget.connect(
            "valueChanged(double)", self.onStepSliderValueChanged
        )
        # Segment color should trigger parameter node modified event
        self.ui.segmentColorPickerButton.connect(
            "colorChanged(QColor)", self._onParameterNodeModified
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
        self.ui.record4DPushButton.disconnect(
            "clicked(bool)", self.onRecord4DButtonClicked
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
        pn = self._parameterNode
        pn.lightingMode = LightingModeEnum.POINT_HEADLIGHT
        pn.layoutSelection = LayoutSelectionEnum.DUAL_3D

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

        # self.updatingParameterNode = True
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

        # Set flythrough frame number slider maximum to match sequence browser number of frames
        if pn.sequenceBrowser:
            self.ui.browserFrameForFlythroughSlider.enabled = True
            self.ui.browserFrameForFlythroughSlider.maximum = (
                pn.sequenceBrowser.GetNumberOfItems() - 1
            )
            # Also set the current frame to this value
            pn.sequenceBrowser.SetSelectedItemNumber(int(pn.browserFrameForFlythrough))
        else:
            # No browser selected
            self.ui.browserFrameForFlythroughSlider.maximum = 10
            self.ui.browserFrameForFlythroughSlider.enabled = False

        # Set playback timer interval
        self.timer.setInterval(pn.playbackTimerIntervalMilliseconds)
        # Ensure that cycleStepsString is valid before enabling record 4D video button
        try:
            cycleSteps = self.logic.getCycleStepsFromString(pn.cycleStepsString)
            cycleStepsStringValidFlag = True
            self.logic.cycleSteps = cycleSteps
        except CycleStepsStringConversionError:
            cycleStepsStringValidFlag = False
        enableRecord4DBool = (
            cycleStepsStringValidFlag and pn.sequenceBrowser and enableRecordBool
        )
        if enableRecord4DBool:
            self.ui.record4DPushButton.enabled = True
            self.ui.record4DPushButton.toolTip = "Record 4D video to file"
        else:
            self.ui.record4DPushButton.enabled = False
            tips = ["To enable:"]
            if not cycleStepsStringValidFlag:
                tips.append('Enter a valid value for "Steps to Show Cycle At"')
            if not pn.sequenceBrowser:
                tips.append("Select a sequence browser")
            if not enableRecordBool:
                if (
                    not pn.cameraLocationsCurveNode
                    or not pn.cameraLocationsCurveNode.GetNumberOfControlPoints() > 1
                ):
                    tips.append(
                        "Camera locations curve must be specified and must have more than 1 control point"
                    )
                if not pn.focalPointsCurveNode:
                    tips.append("Focal points curve must be specified")
                if not pn.videoSaveFilePath:
                    tips.append("Video save path must be specified")
            tipStr = "\n".join(tips)
            self.ui.record4DPushButton.toolTip = tipStr
        # Handle segment color picker color selection (pass on to logic property)
        segQColor = self.ui.segmentColorPickerButton.color
        r, g, b = (segQColor.redF(), segQColor.greenF(), segQColor.blueF())
        self.logic.segmentColor = [r, g, b]
        # Control cone visiblity
        if pn.coneModel:
            pn.coneModel.GetDisplayNode().SetVisibility(pn.addCameraConeFlag)
        # Keep an eye on whether this is getting called properly (or if it is getting double-called)
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
        if pn.addCameraConeFlag:
            if not pn.coneModel:
                pn.coneModel = self.logic.createConeModel()
                pn.coneTransform = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLLinearTransformNode", "ConeTransform"
                )
                pn.coneModel.SetAndObserveTransformNodeID(pn.coneTransform.GetID())
            # Update transform to new camera position
            if pn.cameraNode is None:
                cameraNode = self.logic.getDefaultCameraNode()
            else:
                cameraNode = pn.cameraNode
            self.logic.updateConeModelToCameraTranform(cameraNode, pn.coneTransform)
        # print(f"New step is {stepValue}")

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
        self.ui.playPushButton.checked = False

    def onRecordPushButtonClicked(self) -> None:
        """Record video of full flythrough"""
        # Gather parameters
        # Capture the video
        self.logic.recordVideo(self._parameterNode)

    def onRecord4DButtonClicked(self) -> None:
        """Record video of full flythrough with dynamic cycling
        at indicated steps
        """
        pn = self._parameterNode
        self.logic.record4DVideo(pn)

    def onPreprocessButtonClicked(self) -> None:
        """
        Preprocess input curve to camera locations and focal points
        """
        # Gather inputs
        inputCurveNode = self._parameterNode.inputCurve
        useSmoothing = self._parameterNode.useSmoothingBool
        smoothingWindow = self._parameterNode.smoothingWindow
        smoothingOrder = self._parameterNode.smoothingOrder
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
            useSmoothing,
            smoothingWindow,
            smoothingOrder,
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
        # There are a few parameters which don't play well with the
        # parameter node wrapper.  Instead they will be derived and stored
        # in the logic object.  These will include cycleSteps (a list of
        # integers derived from a string), and the segment color choice
        # (chosen using a widget that can't interface with the wrapper).
        self.cycleSteps = []
        self.segmentColor = []

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

    def getQThreeDViewByViewName(self, name="View1") -> qMRMLThreeDView:
        """Note that this returns something slightly different than
        getThreeDViewNodeByName().  This one is the one to use when
        you need to get the renderer for the view.  The other one
        is when you need the mrmlViewNode (like for screen capture
        module use).
        """
        lm = slicer.app.layoutManager()
        for viewIdx in range(lm.threeDViewCount):
            threeDView = lm.threeDWidget(viewIdx).threeDView()
            viewNode = lm.threeDWidget(viewIdx).mrmlViewNode()
            if viewNode.GetName() == name:
                return threeDView
        # Name failed to match any 3D view
        raise ViewNameNotFoundError(f'View named "{name}" not found!')

    def getCycleStepsFromString(self, cycleStepsStr: str):
        """Process string into list of integer step numbers to run the
        dynamic cycle at. Split on commas, then in each group of characters,
        strip away non-digit characters from either end, then convert to int.
        If this conversion fails at any of the split values
        """
        cycleSteps = []
        for cycleStepStr in cycleStepsStr.split(","):
            # Strip leading and trailing non-digit characters
            cycleStepStrTrimmed = re.sub(r"^\D+|\D+$", "", cycleStepStr)
            try:
                cycleStep = int(cycleStepStrTrimmed)
            except ValueError:
                raise CycleStepsStringConversionError(
                    f"Could not convert '{cycleStepStr} to integer!"
                )
            cycleSteps.append(cycleStep)
        if len(cycleSteps) == 0:
            raise CycleStepsStringConversionError(
                "No cycle steps found, at least one must be specified for 4D videos!"
            )
        return cycleSteps

    def prepareForRecording(
        self, parameterNode: VirtualEndoscopyParameterNode, fourDFlag=False
    ):
        """Prepare the layout, lighting, etc according to settings."""
        pn = parameterNode
        ## Layout
        if pn.useForceLayoutSelection:
            if pn.layoutSelection == LayoutSelectionEnum.UNCHANGED:
                pass
            else:
                # LayoutSelectionEnum values are layout numbers and can be used directly
                layoutNumber = pn.layoutSelection.value
                slicer.app.layoutManager().setLayout(layoutNumber)

        ## Lighting
        if pn.lightingMode == LightingModeEnum.POINT_HEADLIGHT:
            self.setUpDefaultEndoLighting()
        elif pn.lightingMode == LightingModeEnum.DIRECTION_HEADLIGHT:
            self.setUpDirectionalHeadlight()
        elif pn.lightingMode == LightingModeEnum.UNCHANGED:
            pass
        else:
            raise (Exception(f"Unknown lighting mode supplied!"))

        ## 4D Segment Color
        if fourDFlag and pn.sequenceBrowser and pn.useStandardize4DSegmentColor:
            browser = pn.sequenceBrowser
            segOutputs = self.getSegmentationNodesFromBrowser(browser)
            nSegNodes = len(segOutputs)
            if nSegNodes == 0:
                warningDisplay(
                    "Can't set 4D segment color because there are no segmentation nodes associated with the browser!"
                )
            elif nSegNodes > 1:
                warningDisplay(
                    "Multiple segmentation nodes are associated with the browser, 4D segment color will be set on ALL of them!"
                )

            for proxNode, segSeqNode in segOutputs:
                self.setSeqSegmentColor(browser, proxNode, color=self.segmentColor)

    def setSeqSegmentColor(self, browser, proxNode, color, segmentIdx=0):
        """Cycle through browser and, in each frame, set the color of the
        segmentIdx'th segment to the supplied color.  The color should be
        a list of three rgb values, fractional 0-1.
        """
        for idx in range(browser.GetNumberOfItems()):
            browser.SetSelectedItemNumber(idx)
            segmentID = proxNode.GetSegmentation().GetNthSegmentID(segmentIdx)
            proxNode.GetSegmentation().GetSegment(segmentID).SetColor(color)

    def getSegmentationNodesFromBrowser(self, browserNode):
        """Check all sequence nodes for this browser and return
        the sequence nodes and their proxy nodes which are segmentation
        nodes.
        """
        segOutputs = []
        proxyCollection = vtk.vtkCollection()
        browserNode.GetAllProxyNodes(proxyCollection)
        for proxIdx in range(proxyCollection.GetNumberOfItems()):
            proxNode = proxyCollection.GetItemAsObject(proxIdx)
            if proxNode.IsA("vtkMRMLSegmentationNode"):
                seqNode = browserNode.GetSequenceNode(proxNode)
                segOutputs.append((proxNode, seqNode))
        return segOutputs

    def record4DVideo(
        self,
        parameterNode: VirtualEndoscopyParameterNode,
        ffmpegExtraOptions: str = "-codec libx264 -vf scale=-2:{videoHeight} -pix_fmt yuv420p",
    ):
        """Record a flythrough video like recordVideo, but pause at the
        steps listed in cycleSteps, and cycle through the dynamic frames, then
        continue through the flythrough.  Or, if it was the last cycleStep,
        and the stop after last cycle step checkbox was checked, then end
        the video there.
        """
        self.prepareForRecording(parameterNode, fourDFlag=True)

        captureLogic = ScreenCapture.ScreenCaptureLogic()

        imgFilePaths, imagePattern = self.save4DImageSeriesByParameterNode(
            parameterNode=parameterNode, cycleSteps=self.cycleSteps
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
        if parameterNode.deleteImages:
            for imgFile in imgFilePaths:
                imgFile.unlink()
            try:
                imageDirectory.rmdir()
            except:
                pass

    def recordVideo(
        self,
        parameterNode: VirtualEndoscopyParameterNode,
        ffmpegExtraOptions: str = "-codec libx264 -vf scale=-2:{videoHeight} -pix_fmt yuv420p",
    ):
        """Save flythrough as a series of images and then compile into a video"""
        self.prepareForRecording(parameterNode, fourDFlag=False)

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
        if parameterNode.deleteImages:
            for imgFile in imgFilePaths:
                imgFile.unlink()

    def save4DImageSeriesByParameterNode(
        self,
        parameterNode: VirtualEndoscopyParameterNode,
        cycleSteps: List[int] = None,
        imageDirectory: Optional[pathlib.Path] = None,
        imageFilePattern: str = "tempImage_%05d.png",
    ) -> tuple[list[pathlib.Path], str]:
        """Save image series using parameters from parameter node.
        Return list of saved image file names and the image file pattern used.
        This 4D version adds the capability to show temporal loop repetitions.
        """
        pn = parameterNode
        captureLogic = ScreenCapture.ScreenCaptureLogic()
        # Determine view to capture (3D only or all views)
        if pn.videoRecordingMode == VideoRecordingModeEnum.THREE_D_ONLY:
            viewToCapture = self.getThreeDViewNodeByName()
        elif pn.videoRecordingMode == VideoRecordingModeEnum.ALL_VIEWS:
            viewToCapture = None  # ScreenCapture treats this as 'capture all views'
        else:
            raise ValueError(
                f"Unknown video recording mode {pn.videoRecordingMode} supplied!"
            )
        # Determine Image Directory
        if imageDirectory is None:
            # Create a temporary subdirectory of video directory
            vidSaveFilePath = pn.videoSaveFilePath
            vidSaveDirectory = vidSaveFilePath.parent
            imageDirectory = pathlib.Path.joinpath(vidSaveDirectory, "TempImageDir")
        # Ensure image directory exists for saving
        imageDirectory.mkdir(parents=True, exist_ok=True)

        imageFilePaths = []
        imageNumber = 0  # counter
        # Build sequence of frame numbers for cycles
        browser = pn.sequenceBrowser
        startFrameIdx = int(pn.browserFrameForFlythrough)
        nFrames = browser.GetNumberOfItems()
        fullTimePointList = list(range(nFrames))
        # Start with continuing from the start frame to the end
        timePointIdxList = list(range(startFrameIdx + 1, nFrames))
        # Add full loops (-1 because 1 loop is built in already)
        for idx in range(pn.repetitionCount - 1):
            timePointIdxList.extend(fullTimePointList)
        # Finish with returning to the start frame
        timePointIdxList.extend(list(range(startFrameIdx + 1)))
        ### Loop and gather flythrough and time loop images
        for stepNumber in range(pn.numberOfSteps):
            pn.currentStepIndex = stepNumber
            self.jumpCameraByParameterNode(parameterNode=pn)
            imageFileName = imageFilePattern % (imageNumber)
            imageFilePath = pathlib.Path(imageDirectory, imageFileName)
            captureLogic.captureImageFromView(
                view=viewToCapture, filename=imageFilePath
            )
            imageFilePaths.append(imageFilePath)
            # increment image number
            imageNumber += 1
            if stepNumber in cycleSteps:
                # Add dynamic cycle images here
                for frameIdx in timePointIdxList:
                    browser.SetSelectedItemNumber(frameIdx)
                    # slicer.app.processEvents()  # otherwise plots don't always update in time
                    # NOTE: even with processEvents() plots don't update in time, they only
                    # update every other frame, which is a bit confusing. Not critical at the
                    # moment, but should return here if it is ever critical.
                    # The problem shows up at the image capture level, before ffmpeg is involved
                    imageFileName = imageFilePattern % (imageNumber)
                    imageFilePath = pathlib.Path(imageDirectory, imageFileName)
                    captureLogic.captureImageFromView(
                        view=viewToCapture, filename=imageFilePath
                    )
                    imageFilePaths.append(imageFilePath)
                    # increment image number
                    imageNumber += 1
                # Stop the flythrough early if requested and it's time
                if pn.stopAfterLastCycleFlag and stepNumber == cycleSteps[-1]:
                    break
        return imageFilePaths, imageFilePattern

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
        useSmoothing: bool,
        smoothingWindow: int,
        smoothingOrder: int,
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
        # Smooth if requested
        if useSmoothing:
            self.smoothCurveNode(cameraLocations, smoothingWindow, smoothingOrder)
            if useResampleSpacing:
                # Resample again to ensure uniform spacing after smoothing
                self.resampleCurveNode(
                    cameraLocations, resampleSpacingMm, cameraLocations
                )

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

    def smoothCurveNode(
        self,
        curvePointsNode: vtkMRMLMarkupsNode,
        smoothingWindow: int = 11,
        smoothingOrder: int = 1,
    ):
        """
        Smooths the input curve's control points using a Savitzky-Golay filter
        """
        smooth_curve_savgol(
            curve_node=curvePointsNode,
            window_length=smoothingWindow,
            polyorder=smoothingOrder,
            create_new_node=False,
        )

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

    def getDefaultCameraNode(self):
        """Get camera node associated with View1 as the default"""
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
                return cameraNode
        raise (Exception("Default camera node not found!"))

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
            cameraNode = self.getDefaultCameraNode()
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
        """This function can be called to standardize the color of the airway
        segment, the layout (dual 3D), and lighting (point source light at camera)
        """
        if segmentColor:
            jupyterNbFcns.set_sequence_segment_color(
                sequenceBrowserNode,
                segmentationProxyNode,
                segmentName,
                color=segmentColor,
            )
        else:
            # don't overwrite default color listed in function definition...
            jupyterNbFcns.set_sequence_segment_color(
                sequenceBrowserNode, segmentationProxyNode, segmentName
            )
        # Change layout to Dual3D view if it's not already
        Dual3DLayoutID = 15
        slicer.app.layoutManager().setLayout(Dual3DLayoutID)
        # Set better lighting for endoscopy view
        self.setUpDefaultEndoLighting()
        # jupyterNbFcns.setup_lighting()

    def setUpDefaultEndoLighting(self, intensity=1.2, coneAngle=90, viewName="View1"):
        """Set up the default endoscopy lighting.  This is a point source light
        located at the camera.
        The default directional headlight is generally too dark on the sides, and
        the set of directional lights in a lightKit also doesn't work well (shadows
        in weird places). Both problems are because the general tube shape of
        lumens have surfaces mostly perpendicular to the directional light
        direction(s), and therefore end up poorly lit.
        We could experiment if offsetting the light from the exact camera center
        enhances things, but it could also potentially cause issues (e.g. if the
        light ended up outside the lumen while the camera was inside but near the
        edge). Having it at the camera center at least ensures the scene is
        illuminated.
        """
        threeDView = self.getQThreeDViewByViewName(viewName)
        renderWindow = threeDView.renderWindow()
        renderer = renderWindow.GetRenderers().GetFirstRenderer()
        # Remove existing lights
        lights = renderer.GetLights()
        lights.InitTraversal()
        light = lights.GetNextItem()
        while light:
            renderer.RemoveLight(light)
            light = lights.GetNextItem()
        # Create new point source light
        light = vtk.vtkLight()
        light.SetLightTypeToCameraLight()  # Attach the light to the camera
        light.SetPositional(True)  # Make the light a point source
        light.SetConeAngle(90)
        light.SetIntensity(1.5)
        # Could also control light color here
        # Could also control ambient, diffuse, specular light colors

        # Add to renderer and re-render
        renderer.AddLight(light)
        renderWindow.Render()

    def updateConeModelToCameraTranform(self, cameraNode, transformNode):
        """ """

        # Get camera position and focal point
        camera_position = np.array(cameraNode.GetPosition())
        focal_point = np.array(cameraNode.GetFocalPoint())

        # Compute the camera direction vector
        direction = focal_point - camera_position
        direction /= np.linalg.norm(direction)

        # Define the default up vector (positive Z-axis)
        camUp = np.array([0, 0, 1])

        # Check if the direction is close to the positive Z-axis
        if np.abs(np.dot(direction, camUp)) > 0.9:
            # Switch to a different up vector (positive Y-axis)
            camUp = np.array([0, 1, 0])

        # Compute the binormal vector using cross product
        camRight = np.cross(camUp, direction)
        camRight /= np.linalg.norm(camRight)

        # Recompute the up vector to ensure orthogonality
        camUp = np.cross(direction, camRight)

        # Create the transformation matrix
        transformMatrix = np.eye(4)
        transformMatrix[:3, 0] = direction
        transformMatrix[:3, 1] = camUp
        transformMatrix[:3, 2] = camRight
        transformMatrix[:3, 3] = camera_position
        # Update transform node from matrix
        slicer.util.updateTransformMatrixFromArray(transformNode, transformMatrix)

    def createConeModel(
        self,
        height=10,
        radius=10,
        resolution=50,
        capping=0,
        color=(0.54, 0.54, 0.54),
        opacity=0.5,
    ):
        """
        This function creates a cone which has its tip at the origin and opens in the
        positive R direction.
        VTK default Cones are created with direction opening to the left (-R) and
        with center (halfway along the height of the cone) at the origin. That puts the
        point of the cone at (H/2,0,0), so this funciton reflects and translates to
        satisfy having the tip at the origin and opening towards the right.
        """

        cone = vtk.vtkConeSource()
        cone.SetCenter(height / 2, 0, 0)
        cone.SetDirection(-1, 0, 0)  # Opens towards the positive R axis
        cone.SetHeight(height)
        cone.SetRadius(radius)
        cone.SetResolution(resolution)
        cone.SetCapping(capping)
        cone.Update()

        coneModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "Cone")
        coneModel.SetAndObservePolyData(cone.GetOutput())
        coneModel.CreateDefaultDisplayNodes()
        # Display properties
        dn = coneModel.GetDisplayNode()
        dn.SetColor(*color)
        dn.SetOpacity(opacity)
        dn.SetBackfaceCulling(0)
        dn.SetViewNodeIDs(
            ["vtkMRMLViewNode2"]
        )  # only visible in View2 (so it doesn't obscure the View1 camera)
        return coneModel

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


class ViewNameNotFoundError(Exception):
    pass


class CycleStepsStringConversionError(Exception):
    pass


# MARK: Helper functions
# Curve smoothing function using Savitzky-Golay filter
from scipy.signal import savgol_filter


def smooth_curve_savgol(
    curve_node, window_length=11, polyorder=1, create_new_node=False
):
    """
    Smooths the control points of a vtkMRMLMarkupsCurveNode using a
    Savitzky-Golay filter.

    Args:
        curve_node (vtkMRMLMarkupsCurveNode): The curve node to be smoothed.
        window_length (int): The length of the filter window (i.e., the number of
                             coefficients). window_length must be a positive odd integer.
        polyorder (int): The order of the polynomial used to fit the samples.
                         polyorder must be less than window_length.
        create_new_node (bool): If True, a new curve node is created and returned
                                with the smoothed control points. If False, the
                                input curve_node is modified in place.

    Returns:
        vtkMRMLMarkupsCurveNode: The smoothed curve node. This will be a new
                                 node if create_new_node is True, otherwise it
                                 will be the input node.
    """
    # Check for valid window_length and polyorder
    if window_length % 2 == 0 or window_length <= 0:
        raise ValueError("window_length must be a positive odd integer.")
    if polyorder >= window_length:
        raise ValueError("polyorder must be less than window_length.")

    # Get the control points as a vtkPoints object
    point_array = arrayFromMarkupsControlPoints(curve_node)
    num_points = point_array.shape[0]

    if num_points < window_length:
        print(
            f"Warning: Not enough points ({num_points}) for the specified window length ({window_length}). No smoothing applied."
        )
        # Return the original node without modification
        if create_new_node:
            new_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode")
            new_node.Copy(curve_node)
            return new_node
        else:
            return curve_node

    # Apply the Savitzky-Golay filter to each coordinate column
    smoothed_array = savgol_filter(
        point_array, window_length, polyorder, axis=0, mode="nearest"
    )
    # Create or update the curve node with the smoothed points
    if create_new_node:
        # Create a new curve node and set the smoothed points
        smoothed_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode")
        smoothed_node.SetName(f"{curve_node.GetName()}_smoothed")
        # Copy other properties from the original node
        smoothed_node.SetAttribute("Slicer_Smoothing_Source", curve_node.GetID())
        # Set control points
        updateMarkupsControlPointsFromArray(smoothed_node, smoothed_array)
        return smoothed_node
    else:
        # Update the existing curve node's points
        updateMarkupsControlPointsFromArray(curve_node, smoothed_array)
        return curve_node
