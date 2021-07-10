import os, sys, time, bpy
from os.path import join, dirname, exists, abspath, split
from mathutils import Vector, Matrix

import numpy as np
import pickle
import glob
import threading
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from mathutils import Vector


from PyQt5.QtWidgets import (
    QWidget,
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QScrollArea,
)

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigCanvas,
    NavigationToolbar2QT as NabToolbar,
)

# Make sure that we are using QT5
matplotlib.use("Qt5Agg")

from scipy.signal import find_peaks
from scipy import signal

import gpu
from gpu_extras.batch import batch_for_shader
import bgl
import blf

import cv2
import cv2.aruco as aruco

# Addon Imports :
from .BDENTAL_4D_Utils import *

addon_dir = dirname(dirname(abspath(__file__)))

#######################################################################################
########################### BDJawTracker Operators ##############################
#######################################################################################

#######################################################################################
# Set UpJaw Operator :
#######################################################################################
class BDENTAL_4D_OT_SetUpJaw(bpy.types.Operator):
    """ will named UpJaw """

    bl_idname = "bdental4d.setupjaw"
    bl_label = "Pick Upper Jaw STL"

    def execute(self, context):

        UpJaw = bpy.context.active_object

        if UpJaw:
            UpJaw.name = "UpJaw"
            message = [
                " DONE!",
            ]
            ShowMessageBox(message=message, icon="COLORSET_03_VEC")
            #bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_VOLUME", center="MEDIAN")
            return {"FINISHED"}
        else:
            message = [
                " Pick Upper Jaw STL!",
            ]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")
            return {"CANCELLED"}


#######################################################################################
# Set UpJaw Operator :
#######################################################################################
class BDENTAL_4D_OT_SetLowJaw(bpy.types.Operator):
    """ will named LowJaw """

    bl_idname = "bdental4d.setlowjaw"
    bl_label = "Pick Lower Jaw STL"

    def execute(self, context):

        LowJaw = bpy.context.active_object
        bpy.context.view_layer.objects.active = LowJaw
        UpJaw = bpy.data.objects["UpJaw"]

        if LowJaw:
            LowJaw.name = "LowJaw"
            message = [
                " DONE!",
            ]
            ShowMessageBox(message=message, icon="COLORSET_03_VEC")
            #bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_VOLUME", center="MEDIAN")
            #bpy.ops.object.constraint_add(type="CHILD_OF")
            #bpy.context.object.constraints["Child Of"].target = UpJaw
            #bpy.context.object.constraints["Child Of"].name = "UpJaw_Child"

            return {"FINISHED"}
        else:
            message = [
                " Pick Lower Jaw STL!",
            ]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")
            return {"CANCELLED"}


#######################################################################################
# AddBoards Operator :
#######################################################################################
class BDENTAL_4D_OT_AddBoards(bpy.types.Operator):
    """ will Add meshes of boards with markers """

    bl_idname = "bdental4d.addboards"
    bl_label = "Add Boards with Markers"

    def execute(self, context):

        # set scene units
        Units = bpy.context.scene.unit_settings
        Units.system = "METRIC"
        Units.scale_length = 0.001
        Units.length_unit = "MILLIMETERS"

        addon_dir = dirname(dirname(abspath(__file__)))
        file_path = join(addon_dir, "Resources", "BlendData", "boards.blend")

        directory = join(file_path, "Object")
        filenameUp = "UpMarker"
        filenameLow = "LowMarker"

        # Add Boards

        bpy.ops.wm.append(filename=filenameUp, directory=directory)
        UpMarker = bpy.data.objects["UpMarker"]
        MoveToCollection(UpMarker, "Markers")
        bpy.ops.object.select_all(action="DESELECT")
        UpMarker.select_set(True)
        bpy.context.view_layer.objects.active = UpMarker
        bpy.ops.object.modifier_add(type="REMESH")
        bpy.context.object.modifiers["Remesh"].mode = "SHARP"
        bpy.context.object.modifiers["Remesh"].octree_depth = 8
        bpy.ops.object.modifier_apply(modifier="Remesh")

        bpy.ops.wm.append(filename=filenameLow, directory=directory)
        LowMarker = bpy.data.objects["LowMarker"]
        MoveToCollection(LowMarker, "Markers")
        LowMarker.select_set(True)
        bpy.context.view_layer.objects.active = LowMarker
        bpy.ops.object.modifier_add(type="REMESH")
        bpy.context.object.modifiers["Remesh"].mode = "SHARP"
        bpy.context.object.modifiers["Remesh"].octree_depth = 8
        bpy.ops.object.modifier_apply(modifier="Remesh")

        Type = "PLAIN_AXES"
        radius = 10
        AddEmpty(
            Type, "Right Condyle", (-50, 47, -76), radius, CollName="Emptys Collection"
        )
        AddEmpty(
            Type, "Left Condyle", (50, 47, -76), radius, CollName="Emptys Collection"
        )
        AddEmpty(Type, "Incisal", (0, -3, 0), radius, CollName="Emptys Collection")
        bpy.ops.object.select_all(action="DESELECT")

        LowMarker = bpy.data.objects["LowMarker"]
        Coll = bpy.data.collections.get("Emptys Collection")
        CollObjects = Coll.objects
        for obj in CollObjects:
            obj.parent = LowMarker

        return {"FINISHED"}


#######################################################################################

#######################################################################################
# Calibration Operator :
#######################################################################################
class BDENTAL_4D_OT_Calibration(bpy.types.Operator):
    """ will check for user camera Calibration file or make new one """

    bl_idname = "bdental4d.calibration"
    bl_label = "Start Calibration"

    def execute(self, context):

        BDENTAL_4D_Props = context.scene.BDENTAL_4D_Props

        # ChAruco board variables
        CHARUCOBOARD_ROWCOUNT = 7
        CHARUCOBOARD_COLCOUNT = 5
        ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_1000)

        # Create constants to be passed into OpenCV and Aruco methods
        CHARUCO_BOARD = aruco.CharucoBoard_create(
            squaresX=CHARUCOBOARD_COLCOUNT,
            squaresY=CHARUCOBOARD_ROWCOUNT,
            squareLength=BDENTAL_4D_Props.UserSquareLength,
            markerLength=BDENTAL_4D_Props.UserMarkerLength,
            dictionary=ARUCO_DICT,
        )

        # Create the arrays and variables we'll use to store info like corners and IDs from images processed
        corners_all = []  # Corners discovered in all images processed
        ids_all = []  # Aruco ids corresponding to corners discovered
        image_size = None  # Determined at runtime

        # This requires a set of images or a video taken with the camera you want to calibrate
        # I'm using a set of images taken with the camera with the naming convention:
        # 'camera-pic-of-charucoboard-<NUMBER>.jpg'
        # All images used should be the same size, which if taken with the same camera shouldn't be a problem
        # images = BDJawTracker_Props.JTrack_UserProjectDir
        # images = glob.glob(AbsPath(BDENTAL_4D_Props.JTrack_UserProjectDir+'*.*')
        images = glob.glob(AbsPath(join(BDENTAL_4D_Props.CalibImages, "*")))
        # Loop through images glob'ed
        if images:

            for iname in images:
                # Open the image
                try:
                    img = cv2.imread(iname)

                    if not img is None:

                        # Grayscale the image
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                        # Find aruco markers in the query image
                        corners, ids, _ = aruco.detectMarkers(
                            image=gray, dictionary=ARUCO_DICT
                        )

                        # Outline the aruco markers found in our query image
                        img = aruco.drawDetectedMarkers(image=img, corners=corners)

                        # Get charuco corners and ids from detected aruco markers
                        (
                            response,
                            charuco_corners,
                            charuco_ids,
                        ) = aruco.interpolateCornersCharuco(
                            markerCorners=corners,
                            markerIds=ids,
                            image=gray,
                            board=CHARUCO_BOARD,
                        )

                        # If a Charuco board was found, let's collect image/corner points
                        # Requiring at least 20 squares
                        if response > 20:
                            # Add these corners and ids to our calibration arrays
                            corners_all.append(charuco_corners)
                            ids_all.append(charuco_ids)

                            # Draw the Charuco board we've detected to show our calibrator the board was properly detected
                            img = aruco.drawDetectedCornersCharuco(
                                image=img,
                                charucoCorners=charuco_corners,
                                charucoIds=charuco_ids,
                            )

                            # If our image size is unknown, set it now
                            if not image_size:
                                image_size = gray.shape[::-1]

                            # Reproportion the image, maxing width or height at 1000
                            proportion = max(img.shape) / 1000.0
                            img = cv2.resize(
                                img,
                                (
                                    int(img.shape[1] / proportion),
                                    int(img.shape[0] / proportion),
                                ),
                            )
                            # Pause to display each image, waiting for key press
                            cv2.imshow("Charuco board", img)
                            cv2.waitKey(1000)
                            print(f"read {split(iname)[1]} ")
                        else:
                            print(
                                "Not able to detect a charuco board in image: {}".format(
                                    split(iname)[1]
                                )
                            )
                except Exception:
                    print(f"Can't read {split(iname)[1]}")
                    pass
            # Destroy any open CV windows
            cv2.destroyAllWindows()

        # Make sure at least one image was found
        else:
            message = [
                "Calibration was unsuccessful!",
                "No valid Calibration images found,",
                "Retry with differents Calibration Images.",
            ]
            ShowMessageBox(message=message, title="INFO", icon="COLORSET_01_VEC")
            print(message)

            # Calibration failed because there were no images, warn the user
            # print(
            #     "Calibration was unsuccessful. No images of charucoboards were found. Add images of charucoboards and use or alter the naming conventions used in this file."
            # )
            # Exit for failure
            return {"CANCELLED"}

        # Make sure we were able to calibrate on at least one charucoboard by checking
        # if we ever determined the image size
        if not image_size:
            # message = "Calibration was unsuccessful. We couldn't detect charucoboards in any of the images supplied. Try changing the patternSize passed into Charucoboard_create(), or try different pictures of charucoboards."
            message = [
                "Calibration was unsuccessful!",
                "Retry with differents Calibration Images.",
            ]
            ShowMessageBox(message=message, title="INFO", icon="COLORSET_01_VEC")
            # Calibration failed because we didn't see any charucoboards of the PatternSize used
            print(message)
            # Exit for failure
            return {"CANCELLED"}

        # Now that we've seen all of our images, perform the camera calibration
        # based on the set of points we've discovered
        (
            calibration,
            cameraMatrix,
            distCoeffs,
            rvecs,
            tvecs,
        ) = aruco.calibrateCameraCharuco(
            charucoCorners=corners_all,
            charucoIds=ids_all,
            board=CHARUCO_BOARD,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None,
        )

        # Print matrix and distortion coefficient to the console
        print(BDENTAL_4D_Props.UserSquareLength)
        print(BDENTAL_4D_Props.UserMarkerLength)
        print(cameraMatrix)
        print(distCoeffs)

        # Save values to be used where matrix+dist is required, for instance for posture estimation
        # I save files in a pickle file, but you can use yaml or whatever works for you
        CalibFile = AbsPath(
            join(BDENTAL_4D_Props.JTrack_UserProjectDir, "calibration.pckl")
        )
        f = open(CalibFile, "wb")
        pickle.dump((cameraMatrix, distCoeffs, rvecs, tvecs), f)
        f.close()

        message = [
            "Calibration was successful!",
            "Calibration file used:",
            CalibFile,
        ]
        ShowMessageBox(message=message, title="INFO", icon="COLORSET_03_VEC")

        # Print to console our success
        print(f"Calibration successful. Calibration file used: {CalibFile}")

        return {"FINISHED"}


#######################################################################################

#######################################################################################
# StarTrack Operator :
#######################################################################################
class BDENTAL_4D_OT_StarTrack(bpy.types.Operator):
    """ will write down tracking data to _DataFile.txt """

    bl_idname = "bdental4d.startrack"
    bl_label = "Start Tracking"

    def execute(self, context):
        BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props
        ProjDir = AbsPath(BDENTAL_4D_Props.JTrack_UserProjectDir)
        CalibFile = join(ProjDir, "calibration.pckl")
        DataFile = join(ProjDir, str(BDENTAL_4D_Props.TrackFile)+"_DataFile.txt")
        TrackFile = AbsPath(BDENTAL_4D_Props.TrackFile)
        #############################################################################################
        # create file and erase :
        with open(DataFile, "w+") as fw:
            fw.truncate(0)

        #############################################################################################
        start = time.perf_counter()
        resize = 1
        # Make a dictionary of {MarkerId : corners}
        MarkersIdCornersDict = dict()
        ARUCO_PARAMETERS = aruco.DetectorParameters_create()
        ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_APRILTAG_36h11)

        if BDENTAL_4D_Props.TrackingType == "Precision":
            ARUCO_PARAMETERS.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
            resize = 1
            ARUCO_PARAMETERS.aprilTagDeglitch = 0
            ARUCO_PARAMETERS.aprilTagMinWhiteBlackDiff = 30
            ARUCO_PARAMETERS.aprilTagMaxLineFitMse = 20
            ARUCO_PARAMETERS.aprilTagCriticalRad = 0.1745329201221466 * 6
            ARUCO_PARAMETERS.aprilTagMinClusterPixels = 5
            ARUCO_PARAMETERS.maxErroneousBitsInBorderRate = 0.35
            ARUCO_PARAMETERS.errorCorrectionRate = 1.0
            ARUCO_PARAMETERS.minMarkerPerimeterRate = 0.05
            ARUCO_PARAMETERS.maxMarkerPerimeterRate = 4
            ARUCO_PARAMETERS.polygonalApproxAccuracyRate = 0.05
            ARUCO_PARAMETERS.minCornerDistanceRate = 0.05
        elif BDENTAL_4D_Props.TrackingType == "Fast":
            ARUCO_PARAMETERS.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
            resize = 2
        elif BDENTAL_4D_Props.TrackingType == "Precision resized(1/2)":
            ARUCO_PARAMETERS.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
            resize = 2
        elif BDENTAL_4D_Props.TrackingType == "Fast resized(1/2)":
            ARUCO_PARAMETERS.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
            resize = 2

        ##############################################################################################
        Board_corners_upper = [
            np.array(
                [
                    [-0.026065, 0.019001, 0.00563],
                    [-0.012138, 0.019001, 0.015382],
                    [-0.012138, 0.001999, 0.015382],
                    [-0.026065, 0.001999, 0.00563],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [-0.0085, 0.019, 0.016528],
                    [0.0085, 0.019, 0.016528],
                    [0.0085, 0.002, 0.016528],
                    [-0.0085, 0.002, 0.016528],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [0.012138, 0.019, 0.015382],
                    [0.026064, 0.019, 0.00563],
                    [0.026064, 0.002, 0.00563],
                    [0.012138, 0.002, 0.015382],
                ],
                dtype=np.float32,
            ),
        ]
        #############################################################################################

        board_corners_lower = [
            np.array(
                [
                    [-0.026064, -0.002, 0.00563],
                    [-0.012138, -0.002, 0.015382],
                    [-0.012138, -0.019, 0.015382],
                    [-0.026064, -0.019, 0.00563],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [-0.0085, -0.002, 0.016528],
                    [0.0085, -0.002, 0.016528],
                    [0.0085, -0.019, 0.016528],
                    [-0.0085, -0.019, 0.016528],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [0.012138, -0.001999, 0.015382],
                    [0.026065, -0.001999, 0.00563],
                    [0.026065, -0.019001, 0.00563],
                    [0.012138, -0.019001, 0.015382],
                ],
                dtype=np.float32,
            ),
        ]

        #############################################################################################

        # Initiate 2 Bords LowBord and UpBoard
        LowBoard_ids = np.array([[3], [4], [5]], dtype=np.int32)
        LowBoard = aruco.Board_create(board_corners_lower, ARUCO_DICT, LowBoard_ids)
        UpBoard_ids = np.array([[0], [1], [2]], dtype=np.int32)
        UpBoard = aruco.Board_create(Board_corners_upper, ARUCO_DICT, UpBoard_ids)

        ##############################################################################################
        if not exists(TrackFile):
            message = [" Invalid Track file check and retry."]
            ShowMessageBox(message=message, icon="COLORSET_01_VEC")
            return {"CANCELLED"}

        if not exists(CalibFile):
            message = [
                "calibration.pckl not found in project directory check and retry."
            ]
            ShowMessageBox(message=message, icon="COLORSET_01_VEC")
            return {"CANCELLED"}

        with open(CalibFile, "rb") as rf:
            (cameraMatrix, distCoeffs, _, _) = pickle.load(rf)
            if cameraMatrix is None or distCoeffs is None:
                message = [
                    "Invalid Calibration File.",
                    "Please replace calibration.pckl",
                    "or recalibrate the camera.",
                ]
                ShowMessageBox(message=message, icon="COLORSET_01_VEC")
                return {"CANCELLED"}

        cap = cv2.VideoCapture(TrackFile)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        totalstr = str(total)
        count = 1
        #############################################################################################
        # Text parameters
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (150, 150)
        org1 = (150, 250)
        org2 = (150, 50)
        # fontScale
        fontScale = 2
        fontScaleSmall = 1
        # Blue color in BGR
        color = (0, 255, 228)
        # Line thickness of 2 px
        thickness = 4
        thickness1 = 2
        fps = format(cap.get(cv2.CAP_PROP_FPS), ".2f")
        height = str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = str(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        print(cameraMatrix)
        print(distCoeffs)
        print(BDENTAL_4D_Props.TrackingType)
        #############################################################################################
        Data_dict = {
            "Width": width,
            "Heihgt": height,
            "Fps": fps,
            "TrackingType": BDENTAL_4D_Props.TrackingType,
            "Stream": {},
        }
        while True:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
            success, img = cap.read()
            if success:
                imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                if resize == 2:
                    imgGrey = cv2.pyrDown(imgGrey)  # reduce the image by 2 times

                cv2.namedWindow("img", cv2.WINDOW_NORMAL)

                # lists of ids and the corners beloning to each id
                corners, ids, rejected = aruco.detectMarkers(
                    imgGrey,
                    ARUCO_DICT,
                    parameters=ARUCO_PARAMETERS,
                    cameraMatrix=cameraMatrix,
                    distCoeff=distCoeffs,
                )

                # Require 2> markers before drawing axis
                if ids is not None and len(ids) >= 6:
                    for i in range(len(ids)):
                        MarkersIdCornersDict[ids[i][0]] = (list(corners))[i]

                    LowCorners = [
                        MarkersIdCornersDict[3],
                        MarkersIdCornersDict[4],
                        MarkersIdCornersDict[5],
                    ]
                    UpCorners = [
                        MarkersIdCornersDict[0],
                        MarkersIdCornersDict[1],
                        MarkersIdCornersDict[2],
                    ]

                    # Estimate the posture of the board, which is a construction of 3D space based on the 2D video
                    Lowretval, Lowrvec, Lowtvec = cv2.aruco.estimatePoseBoard(
                        LowCorners,
                        LowBoard_ids,
                        LowBoard,
                        cameraMatrix / resize,
                        distCoeffs,
                        None,
                        None,
                    )
                    Upretval, Uprvec, Uptvec = cv2.aruco.estimatePoseBoard(
                        UpCorners,
                        UpBoard_ids,
                        UpBoard,
                        cameraMatrix / resize,
                        distCoeffs,
                        None,
                        None,
                    )

                    if Lowretval and Upretval:
                        # Draw the camera posture calculated from the board
                        Data_dict["Stream"][count] = {
                            "UpBoard": [
                                (Uptvec[0, 0], Uptvec[1, 0], Uptvec[2, 0]),
                                (Uprvec[0, 0], Uprvec[1, 0], Uprvec[2, 0]),
                            ],
                            "LowBoard": [
                                (Lowtvec[0, 0], Lowtvec[1, 0], Lowtvec[2, 0]),
                                (Lowrvec[0, 0], Lowrvec[1, 0], Lowrvec[2, 0]),
                            ],
                        }
                        count += 1
                        img = aruco.drawAxis(
                            img,
                            cameraMatrix,
                            distCoeffs,
                            Uprvec,
                            Uptvec,
                            0.05,
                        )
                        img = aruco.drawAxis(
                            img,
                            cameraMatrix,
                            distCoeffs,
                            Lowrvec,
                            Lowtvec,
                            0.05,
                        )

                        currentFrame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        currentFramestr = str(currentFrame)
                        perсent = currentFrame / total * 100
                        perсent = str("{0:.2f}%".format(perсent))

                        img = cv2.putText(
                            img,
                            currentFramestr + " frame of " + totalstr,
                            org,
                            font,
                            fontScale,
                            color,
                            thickness,
                            cv2.LINE_AA,
                        )
                        img = cv2.putText(
                            img,
                            perсent,
                            org1,
                            font,
                            fontScale,
                            color,
                            thickness,
                            cv2.LINE_AA,
                        )
                        img = cv2.putText(
                            img,
                            'to stop tracking press "Q"',
                            org2,
                            font,
                            fontScaleSmall,
                            color,
                            thickness1,
                            cv2.LINE_AA,
                        )

                        if resize == 2:
                            img = cv2.pyrDown(img)
                            img = cv2.aruco.drawDetectedMarkers(
                                img, corners, ids, (0, 255, 0)
                            )

                        else:
                            img = cv2.aruco.drawDetectedMarkers(
                                img, corners, ids, (0, 255, 0)
                            )

                        cv2.imshow("img", img)

                        if currentFrame == total:
                            cv2.destroyAllWindows()
                            break

        with open(DataFile, "a") as DF:
            Data = str(Data_dict)
            DF.write(Data)
        BDENTAL_4D_Props.TrackedData = RelPath(DataFile)

        return {"FINISHED"}


#######################################################################################

#######################################################################################
# Data reading operator :
#######################################################################################
class BDENTAL_4D_OT_DataReader(bpy.types.Operator):
    """ Data reading and Transfert to Blender animation """

    bl_idname = "bdental4d.datareader"
    bl_label = "Start Read Data"

    def execute(self, context):
        BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props

        def DataToCvMatrix(DataFile):

            UpCvMatrix_List, LowCvMatrix_List = [], []

            with open(DataFile, "r") as DataRead:
                Data = DataRead.read()
                Data_dict = eval(Data)

            width = Data_dict["Width"]
            heihgt = Data_dict["Heihgt"]
            fps = Data_dict["Fps"]
            TrackingType = Data_dict["TrackingType"]
            Stream = Data_dict["Stream"]

            for k, v in Stream.items():
                FrameNumber = k
                FrameData = v

                UpBoard_tvec, UpBoard_rvec = (
                    Vector(FrameData["UpBoard"][0]) * 1000,
                    FrameData["UpBoard"][1],
                )
                LowBoard_tvec, LowBoard_rvec = (
                    Vector(FrameData["LowBoard"][0]) * 1000,
                    FrameData["LowBoard"][1],
                )

                UpRotMat, LowRotMat = Matrix(cv2.Rodrigues(UpBoard_rvec)[0]), Matrix(
                    cv2.Rodrigues(LowBoard_rvec)[0]
                )

                UpCvMatrix = UpRotMat.to_4x4()
                UpCvMatrix[0][3], UpCvMatrix[1][3], UpCvMatrix[2][3] = UpBoard_tvec
                UpCvMatrix_List.append(UpCvMatrix)

                LowCvMatrix = LowRotMat.to_4x4()
                LowCvMatrix[0][3], LowCvMatrix[1][3], LowCvMatrix[2][3] = LowBoard_tvec
                LowCvMatrix_List.append(LowCvMatrix)

            TotalFrames = len(UpCvMatrix_List)

            return (
                width,
                heihgt,
                fps,
                TrackingType,
                TotalFrames,
                UpCvMatrix_List,
                LowCvMatrix_List,
            )

        def Blender_Matrix(
            UpBoard_Obj,
            LowBoard_Obj,
            UpCvMatrix_List,
            LowCvMatrix_List,
            TotalFrames,
            Stab=False,
        ):

            UpBoard_Aligned = UpBoard_Obj.matrix_world.copy()
            LowBoard_Aligned = LowBoard_Obj.matrix_world.copy()

            Transform = UpBoard_Aligned @ UpCvMatrix_List[0].inverted()

            UpBlender_Matrix_List, LowBlender_Matrix_List = [], []

            for i in range(TotalFrames):
                UpBlender_Matrix = Transform @ UpCvMatrix_List[i]
                LowBlender_Matrix = Transform @ LowCvMatrix_List[i]

                UpBlender_Matrix_List.append(UpBlender_Matrix)
                LowBlender_Matrix_List.append(LowBlender_Matrix)
            if not Stab:
                return UpBlender_Matrix_List, LowBlender_Matrix_List
            else:
                UpStabMatrix_List, LowStabMatrix_List = Stab_Low_function(
                    UpBlender_Matrix_List, LowBlender_Matrix_List, TotalFrames
                )
                return UpStabMatrix_List, LowStabMatrix_List

        def Stab_Low_function(
            UpBlender_Matrix_List, LowBlender_Matrix_List, TotalFrames
        ):

            UpStabMatrix_List, LowStabMatrix_List = [], []
            for i in range(TotalFrames):
                StabTransform = (
                    UpBlender_Matrix_List[0] @ UpBlender_Matrix_List[i].inverted()
                )
                LowStabMatrix = StabTransform @ LowBlender_Matrix_List[i]

                UpStabMatrix_List.append(UpBlender_Matrix_List[0])
                LowStabMatrix_List.append(LowStabMatrix)

            return UpStabMatrix_List, LowStabMatrix_List

        def MatrixToAnimation(i, UpMtxList, lowMtxList, UpBoard_Obj, LowBoard_Obj):
            Offset = 1
            UpBoard_Obj.matrix_world = UpMtxList[i]
            UpBoard_Obj.keyframe_insert("location", frame=i * Offset)
            UpBoard_Obj.keyframe_insert("rotation_quaternion", frame=i * Offset)

            LowBoard_Obj.matrix_world = lowMtxList[i]
            LowBoard_Obj.keyframe_insert("location", frame=i * Offset)
            LowBoard_Obj.keyframe_insert("rotation_quaternion", frame=i * Offset)

            print(f"Keyframe {i} added..")

        def progress_bar(counter, n, Delay):

            X, Y = WindowWidth, WindowHeight = (500, 100)
            BackGround = np.ones((Y, X, 3), dtype=np.uint8) * 255
            Title = "BD Jaw Tracker"
            # Progress bar Parameters :
            maxFill = X - 70
            minFill = 40
            barColor = (50, 200, 0)
            BarHeight = 20
            barUp = Y - 60
            barBottom = barUp + BarHeight
            # Text :
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            fontThikness = 1
            fontColor = (0, 0, 0)
            lineStyle = cv2.LINE_AA

            chunk = (maxFill - 40) / n

            img = BackGround.copy()
            fill = minFill + int(counter * chunk)
            pourcentage = int(counter * 100 / n)
            img[barUp:barBottom, minFill:fill] = barColor

            img = cv2.putText(
                img,
                f"{pourcentage}%",
                (maxFill + 10, barBottom - 8),
                # (fill + 10, barBottom - 10),
                font,
                fontScale,
                fontColor,
                fontThikness,
                lineStyle,
            )

            img = cv2.putText(
                img,
                "Processing...",
                (minFill, barUp - 10),
                font,
                fontScale,
                fontColor,
                fontThikness,
                lineStyle,
            )
            img = cv2.putText(
                img,
                f"Frame {i}/{n}...",
                (minFill, barBottom + 20),
                font,
                fontScale,
                fontColor,
                fontThikness,
                lineStyle,
            )
            cv2.imshow(Title, img)

            cv2.waitKey(Delay)
            # cv2.destroyAllWindows()
            counter += 1

            # if i == n - 1:
            if counter == n:
                img = BackGround.copy()
                img[barUp:barBottom, minFill:maxFill] = (50, 200, 0)
                img = cv2.putText(
                    img,
                    "100%",
                    (maxFill + 10, barBottom - 8),
                    font,
                    fontScale,
                    fontColor,
                    fontThikness,
                    lineStyle,
                )

                img = cv2.putText(
                    img,
                    "Finished.",
                    (minFill, barUp - 10),
                    font,
                    fontScale,
                    fontColor,
                    fontThikness,
                    lineStyle,
                )
                img = cv2.putText(
                    img,
                    f"Total Frames added : {n}",
                    (minFill, barBottom + 20),
                    font,
                    fontScale,
                    fontColor,
                    fontThikness,
                    lineStyle,
                )
                cv2.imshow(Title, img)
                cv2.waitKey(3000)
                cv2.destroyAllWindows()

        ########################################################################################
        ######################################################################################

        DataFile = AbsPath(BDENTAL_4D_Props.TrackedData)  # DataFile

        (
            width,
            heihgt,
            fps,
            TrackingType,
            TotalFrames,
            UpCvMatrix_List,
            LowCvMatrix_List,
        ) = DataToCvMatrix(DataFile)

        # point to the 2 boards here :
        UpBoard_Obj, LowBoard_Obj = (
            bpy.data.objects.get("UpMarker"),
            bpy.data.objects.get("LowMarker"),
        )

        if UpBoard_Obj and LowBoard_Obj:
            UpStabMatrix_List, LowStabMatrix_List = Blender_Matrix(
                UpBoard_Obj,
                LowBoard_Obj,
                UpCvMatrix_List,
                LowCvMatrix_List,
                TotalFrames,
                Stab=True,
            )

            for obj in [UpBoard_Obj, LowBoard_Obj]:
                obj.animation_data_clear()
                obj.rotation_mode = "QUATERNION"

            for i in range(TotalFrames):
                t = threading.Thread(
                    target=MatrixToAnimation,
                    args=[
                        i,
                        UpStabMatrix_List,
                        LowStabMatrix_List,
                        UpBoard_Obj,
                        LowBoard_Obj,
                    ],
                    daemon=True,
                )
                t.start()
                t.join()
                progress_bar(i, TotalFrames, 1)
            Start = 0
            Offset = 1
            bpy.context.scene.frame_start = Start
            End = (TotalFrames - 1) * Offset
            bpy.context.scene.frame_end = End
            bpy.context.scene.frame_current = 0
            bpy.context.scene.render.fps = float(fps)
            print(float(fps))

            return {"FINISHED"}


#######################################################################################

#######################################################################################
# SmoothKeyframes operator :
#######################################################################################
class BDENTAL_4D_OT_SmoothKeyframes(bpy.types.Operator):
    """ will smooth animation curve """

    bl_idname = "bdental4d.smoothkeyframes"
    bl_label = "Smooth keyframes"

    def execute(self, context):

        LowMarker = bpy.data.objects.get("LowMarker")
        bpy.ops.object.select_all(action="DESELECT")
        LowMarker.hide_set(False)
        LowMarker.select_set(True)
        bpy.context.view_layer.objects.active = LowMarker

        current_area = bpy.context.area.type
        layer = bpy.context.view_layer

        # change to graph editor
        bpy.context.area.type = "GRAPH_EDITOR"

        # smooth curves of all selected bones
        bpy.ops.graph.smooth()

        # switch back to original area
        bpy.context.area.type = current_area
        bpy.ops.object.select_all(action="DESELECT")
        message = [" DONE!"]
        ShowMessageBox(message=message, icon="COLORSET_03_VEC")

        return {"FINISHED"}


#######################################################################################
# Draw or redraw motion path operator :
#######################################################################################
class BDENTAL_4D_OT_DrawPath(bpy.types.Operator):
    """ Draw or redraw motion path """

    bl_idname = "bdental4d.drawpath"
    bl_label = "Draw motion path"

    def execute(self, context):
        scene = bpy.context.scene
        bpy.ops.object.paths_calculate(
            start_frame=scene.frame_start, end_frame=scene.frame_end
        )

        return {"FINISHED"}


class BDENTAL_4D_OT_DrawMovements(bpy.types.Operator):
    """ Draw Emptys movements """

    bl_idname = "bdental4d.drawmovements"
    bl_label = "Draw Movements"

    def execute(self, context):
        ProtrusionColor = [0, 0, 1, 1]
        RightColor = [1.0, 0.0, 0.9, 1.0]
        LeftColor = [1.0, 0.7, 0.0, 1.000000]
        OpenColor = [1, 0, 0, 1]

        CollName = "Mandibular Movements"
        IP = bpy.data.objects.get("Incisal")
        RCP = bpy.data.objects.get("Right Condyle")
        LCP = bpy.data.objects.get("Left Condyle")
        if not (IP and RCP and LCP):
            message = [
                " Please ensure Incisal, Right Condyle and Left Condyle",
                "Emptys are present in Emptys Collection",
            ]
            ShowMessageBox(message=message, icon="COLORSET_01_VEC")
            return {"CANCELLED"}

        PointCouplesList = GetPeakPointCouples(IP, RCP, LCP, DIST=50)

        for Couple in PointCouplesList:
            P0_info = Couple[0]
            P1_info = Couple[1]
            P0 = bpy.data.objects.get(P0_info[0])
            P1 = bpy.data.objects.get(P1_info[0])
            if not P0:
                P0 = AddMarkupPoint(
                    name=P0_info[0],
                    color=(0, 0, 0, 1),
                    loc=P0_info[1],
                    Diameter=0.3,
                    CollName=CollName,
                )
            if not P1:
                P1 = AddMarkupPoint(
                    name=P1_info[0],
                    color=(0, 0, 0, 1),
                    loc=P1_info[1],
                    Diameter=0.3,
                    CollName=CollName,
                )
            SegmentName = f"{P0_info[0]} (Segment)"
            if "Protrusion" in SegmentName:
                SegmentColor = ProtrusionColor
            elif "Right" in SegmentName:
                SegmentColor = RightColor
            elif "Left" in SegmentName:
                SegmentColor = LeftColor
            elif "Open" in SegmentName:
                SegmentColor = OpenColor

            AddHookedSegment(
                Points=[P0, P1],
                Name=SegmentName,
                color=SegmentColor,
                thikness=0.2,
                CollName=CollName,
            )
        return {"FINISHED"}


class BDENTAL_4D_OT_RepportPlot(bpy.types.Operator):
    """ Plot JTrack Repport """

    bl_idname = "bdental4d.repport_plot"
    bl_label = "SHOW REPPORT"

    def execute(self, context):
        ImgFolder = join(addon_dir, "Resources", "Images")
        fig = JTrackRepportPlot(ImgFolder)
        app = QApplication.instance()
        if not app:
            app = QApplication(sys.argv)

        window = MyApp(fig)
        window.initUI()
        app.exec_()
        return {"FINISHED"}


#################################################################################################
# Registration :
#################################################################################################

classes = [
    BDENTAL_4D_OT_SetUpJaw,
    BDENTAL_4D_OT_SetLowJaw,
    BDENTAL_4D_OT_AddBoards,
    BDENTAL_4D_OT_Calibration,
    BDENTAL_4D_OT_StarTrack,
    BDENTAL_4D_OT_DataReader,
    BDENTAL_4D_OT_SmoothKeyframes,
    BDENTAL_4D_OT_DrawPath,
    BDENTAL_4D_OT_DrawMovements,
    BDENTAL_4D_OT_RepportPlot,
]


def register():

    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
