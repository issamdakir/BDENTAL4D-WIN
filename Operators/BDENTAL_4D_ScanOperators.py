import os, stat, sys, shutil, math, threading
from math import degrees, radians, pi
import numpy as np
from time import sleep, perf_counter as Tcounter
from queue import Queue
from os.path import join, dirname, abspath, exists, split
from importlib import reload

# Blender Imports :
import bpy
import bmesh
from mathutils import Matrix, Vector, Euler, kdtree
from bpy.props import (
    StringProperty,
    IntProperty,
    FloatProperty,
    EnumProperty,
    FloatVectorProperty,
    BoolProperty,
)
import SimpleITK as sitk
import vtk
import cv2

# try :
#     cv2 = reload(cv2)
# except ImportError :
#     pass
from vtk.util import numpy_support
from vtk import vtkCommand

# Global Variables :

# from . import BDENTAL_4D_Utils
from .BDENTAL_4D_Utils import *

addon_dir = dirname(dirname(abspath(__file__)))
ShadersBlendFile = join(
    addon_dir, "Resources", "BlendData", "BDENTAL_4D_BlendData.blend"
)
ImplantLibraryBlendFile = join(
    addon_dir, "Resources", "BlendData", "NEOBIOTECH_LIBRARY.blend"
)
GpShader = "VGS_Marcos_modified"  # "VGS_Marcos_01" "VGS_Dakir_01"
Wmin = -400
Wmax = 3000
ProgEvent = vtkCommand.ProgressEvent
#######################################################################################
########################### CT Scan Load : Operators ##############################
#######################################################################################
def rmtree(top):
    for root, dirs, files in os.walk(top, topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            os.chmod(filename, stat.S_IWUSR)
            os.remove(filename)
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(top)


# class BDENTAL_4D_OT_Uninstall(bpy.types.Operator):
#     """ Uninstall Addon """

#     bl_idname = "bdental.uninstall"
#     bl_label = "UNINSTALL"

#     def execute(self, context):

#         # Disable Addon :
#         Addon_Enable(AddonName='BDENTAL_4D', Enable=False)
#         try :
#             shutil.rmtree(addon_dir)
#             print('BDENTAL_4D Addon uninstalled successfully.(shutil)')
#         except Exception as Er :
#             print(Er)
#             if sys.platform == 'win32':
#                 try :
#                     rmtree(addon_dir)
#                     if not exists(addon_dir):
#                         print('BDENTAL_4D Addon uninstalled successfully.(rmtree)')
#                     else :
#                         print('BDENTAL_4D Addon could not be uninstalled ! (Folder still exists wthout error)')
#                 except Exception as Er :
#                     print(f'BDENTAL_4D Addon could not be uninstalled ! Error : {Er}')

#                 # try :
#                 #     os.chmod(addon_dir,stat.S_IWUSR)
#                 #     # os.chmod(addon_dir,stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
#                 #     shutil.rmtree(addon_dir)
#                 #     if not exists(addon_dir):
#                 #         print('BDENTAL_4D Addon uninstalled successfully.(os.chmod)')
#                 #     else :
#                 #         print('BDENTAL_4D Addon could not be uninstalled !')
#                 # except Exception as Er :
#                 #     print(Er)
#                 #     print('BDENTAL_4D Addon could not be uninstalled !')


#         return {"FINISHED"}


class BDENTAL_4D_OT_Template(bpy.types.Operator):
    """ Open BDENTAL_4D workspace template """

    bl_idname = "bdental.template"
    bl_label = "OPEN BDENTAL_4D WORKSPACE"

    SaveMainFile: BoolProperty(description="Save Main File", default=False)
    UserProjectDir: StringProperty(
        name="INFO : ",
        default=" Please Restart Blender after",
        description="",
    )

    def execute(self, context):

        CurrentBlendFile = bpy.path.abspath(bpy.data.filepath)
        BlendStartFile = join(
            addon_dir, "Resources", "BlendData", "BlendStartFile.blend"
        )

        # Install or load BDENTAL_4D theme :
        ScriptsPath = dirname(dirname(addon_dir))
        BDENTAL_4D_Theme_installed = join(
            ScriptsPath, "presets", "interface_theme", "BDENTAL_4D_Theme.xml"
        )
        if not exists(BDENTAL_4D_Theme_installed):
            BDENTAL_4D_Theme = join(addon_dir, "Resources", "BDENTAL_4D_Theme.xml")
            bpy.ops.preferences.theme_install(filepath=BDENTAL_4D_Theme)

        bpy.ops.script.execute_preset(
            filepath=BDENTAL_4D_Theme_installed,
            menu_idname="USERPREF_MT_interface_theme_presets",
        )

        # Save Re Open current Project check :
        if self.SaveMainFile:
            if not CurrentBlendFile:
                message = [" Please Save your Project ", "Or uncheck Save Main File"]
                ShowMessageBox(message=message, icon="COLORSET_01_VEC")
                return {"CANCELLED"}
            else:
                bpy.ops.wm.save_mainfile()
                CurrentBlendFile = bpy.path.abspath(bpy.data.filepath)
                reopen = True
        if not self.SaveMainFile:
            reopen = False

        bpy.ops.wm.open_mainfile(filepath=BlendStartFile)
        bpy.context.preferences.inputs.use_auto_perspective = False
        bpy.context.preferences.inputs.use_rotate_around_active = True
        bpy.context.preferences.inputs.use_mouse_depth_navigate = True
        bpy.context.preferences.view.ui_scale = 1.1
        bpy.ops.wm.save_userpref()
        # context.space_data.region_3d.view_perspective = "ORTHO"
        bpy.ops.wm.save_homefile()

        if reopen and CurrentBlendFile:
            bpy.ops.wm.open_mainfile(filepath=CurrentBlendFile)
        return {"FINISHED"}

    def invoke(self, context, event):

        wm = context.window_manager
        return wm.invoke_props_dialog(self)


def GetMaxSerie(UserDcmDir):

    SeriesDict = {}
    Series_reader = sitk.ImageSeriesReader()
    series_IDs = Series_reader.GetGDCMSeriesIDs(UserDcmDir)

    if not series_IDs:

        message = ["No valid DICOM Serie found in DICOM Folder ! "]
        print(message)
        ShowMessageBox(message=message, icon="COLORSET_01_VEC")
        return {"CANCELLED"}

    def GetSerieCount(sID):
        count = len(Series_reader.GetGDCMSeriesFileNames(UserDcmDir, sID))
        SeriesDict[count] = sID

    threads = [
        threading.Thread(
            target=GetSerieCount,
            args=[sID],
            daemon=True,
        )
        for sID in series_IDs
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()
    MaxCount = sorted(SeriesDict, reverse=True)[0]
    MaxSerie = SeriesDict[MaxCount]
    return MaxSerie, MaxCount


def Load_Dicom_funtion(context, q):

    ################################################################################################
    start = Tcounter()
    ################################################################################################
    BDENTAL_4D_Props = context.scene.BDENTAL_4D_Props
    UserProjectDir = AbsPath(BDENTAL_4D_Props.UserProjectDir)
    UserDcmDir = AbsPath(BDENTAL_4D_Props.UserDcmDir)

    ################################################################################################

    if not exists(UserProjectDir):

        message = ["The Selected Project Directory Path is not valid ! "]
        ShowMessageBox(message=message, icon="COLORSET_02_VEC")
        return {"CANCELLED"}

    elif not exists(UserDcmDir):

        message = [" The Selected Dicom Directory Path is not valid ! "]
        ShowMessageBox(message=message, icon="COLORSET_02_VEC")
        return {"CANCELLED"}

    elif not os.listdir(UserDcmDir):
        message = ["No valid DICOM Serie found in DICOM Folder ! "]
        ShowMessageBox(message=message, icon="COLORSET_02_VEC")
        return {"CANCELLED"}

    else:
        # Get Preffix and save file :
        DcmInfoDict = eval(BDENTAL_4D_Props.DcmInfo)
        Preffixs = list(DcmInfoDict.keys())

        for i in range(1, 100):
            Preffix = f"BD{i:03}"
            if not Preffix in Preffixs:
                break

        Split = split(UserProjectDir)
        ProjectName = Split[-1] or Split[-2]
        BlendFile = f"{ProjectName}_CT-SCAN.blend"
        Blendpath = join(UserProjectDir, BlendFile)

        if not exists(Blendpath) or bpy.context.blend_data.filepath == Blendpath:
            bpy.ops.wm.save_as_mainfile(filepath=Blendpath)
        else:
            bpy.ops.wm.save_mainfile()

        # Start Reading Dicom data :
        ######################################################################################
        Series_reader = sitk.ImageSeriesReader()
        MaxSerie, MaxCount = GetMaxSerie(UserDcmDir)
        DcmSerie = Series_reader.GetGDCMSeriesFileNames(UserDcmDir, MaxSerie)

        ##################################### debug_02 ###################################
        debug_01 = Tcounter()
        message = f"MaxSerie ID : {MaxSerie}, MaxSerie Count : {MaxCount} (Time : {round(debug_01-start,2)} secondes)"
        print(message)
        # q.put("Max DcmSerie extracted...")
        ####################################################################################

        # Get StudyInfo :
        reader = sitk.ImageFileReader()
        reader.SetFileName(DcmSerie[0])
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()

        Image3D = sitk.ReadImage(DcmSerie)

        # Get Dicom Info :
        Sp = Spacing = Image3D.GetSpacing()
        Sz = Size = Image3D.GetSize()
        Dims = Dimensions = Image3D.GetDimension()
        Origin = Image3D.GetOrigin()
        Direction = Image3D.GetDirection()

        # calculate Informations :
        D = Direction
        O = Origin
        DirectionMatrix_4x4 = Matrix(
            (
                (D[0], D[1], D[2], 0.0),
                (D[3], D[4], D[5], 0.0),
                (D[6], D[7], D[8], 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        )

        TransMatrix_4x4 = Matrix(
            (
                (1.0, 0.0, 0.0, O[0]),
                (0.0, 1.0, 0.0, O[1]),
                (0.0, 0.0, 1.0, O[2]),
                (0.0, 0.0, 0.0, 1.0),
            )
        )

        VtkTransform_4x4 = TransMatrix_4x4 @ DirectionMatrix_4x4
        P0 = Image3D.TransformContinuousIndexToPhysicalPoint((0, 0, 0))
        P_diagonal = Image3D.TransformContinuousIndexToPhysicalPoint(
            (Sz[0] - 1, Sz[1] - 1, Sz[2] - 1)
        )
        VCenter = (Vector(P0) + Vector(P_diagonal)) * 0.5

        C = VCenter

        TransformMatrix = Matrix(
            (
                (D[0], D[1], D[2], C[0]),
                (D[3], D[4], D[5], C[1]),
                (D[6], D[7], D[8], C[2]),
                (0.0, 0.0, 0.0, 1.0),
            )
        )

        # Set DcmInfo :

        DcmInfo = {
            "UserProjectDir": RelPath(UserProjectDir),
            "Preffix": Preffix,
            "RenderSz": Sz,
            "RenderSp": Sp,
            "PixelType": Image3D.GetPixelIDTypeAsString(),
            "Wmin": Wmin,
            "Wmax": Wmax,
            "Size": Sz,
            "Dims": Dims,
            "Spacing": Sp,
            "Origin": Origin,
            "Direction": Direction,
            "TransformMatrix": TransformMatrix,
            "DirectionMatrix_4x4": DirectionMatrix_4x4,
            "TransMatrix_4x4": TransMatrix_4x4,
            "VtkTransform_4x4": VtkTransform_4x4,
            "VolumeCenter": VCenter,
        }

        tags = {
            "StudyDate": "0008|0020",
            "PatientName": "0010|0010",
            "PatientID": "0010|0020",
            "BirthDate": "0010|0030",
            "WinCenter": "0028|1050",
            "WinWidth": "0028|1051",
        }
        for k, tag in tags.items():

            if tag in reader.GetMetaDataKeys():
                v = reader.GetMetaData(tag)

            else:
                v = ""

            DcmInfo[k] = v
            Image3D.SetMetaData(tag, v)

        ###################################### debug_02 ##################################
        debug_02 = Tcounter()
        message = f"DcmInfo {Preffix} set (Time : {debug_02-debug_01} secondes)"
        print(message)
        # q.put("Dicom Info extracted...")
        ##################################################################################

        #######################################################################################
        # Add directories :
        SlicesDir = join(UserProjectDir, "Slices")
        if not exists(SlicesDir):
            os.makedirs(SlicesDir)
        DcmInfo["SlicesDir"] = RelPath(SlicesDir)

        PngDir = join(UserProjectDir, "PNG")
        if not exists(PngDir):
            os.makedirs(PngDir)

        Nrrd255Path = join(UserProjectDir, f"{Preffix}_Image3D255.nrrd")

        DcmInfo["Nrrd255Path"] = RelPath(Nrrd255Path)

        #######################################################################################
        # set IntensityWindowing  :
        Image3D_255 = sitk.Cast(
            sitk.IntensityWindowing(
                Image3D,
                windowMinimum=Wmin,
                windowMaximum=Wmax,
                outputMinimum=0.0,
                outputMaximum=255.0,
            ),
            sitk.sitkUInt8,
        )

        # Convert Dicom to nrrd file :
        # sitk.WriteImage(Image3D, NrrdHuPath)
        sitk.WriteImage(Image3D_255, Nrrd255Path)

        ################################## debug_03 ######################################
        debug_03 = Tcounter()
        message = f"Nrrd255 Export done!  (Time : {debug_03-debug_02} secondes)"
        print(message)
        # q.put("nrrd 3D image file saved...")
        ##################################################################################

        #############################################################################################
        # MultiThreading PNG Writer:
        #########################################################################################
        def Image3DToPNG(i, slices, PngDir, Preffix):
            img_Slice = slices[i]
            img_Name = f"{Preffix}_img{i:04}.png"
            image_path = join(PngDir, img_Name)
            cv2.imwrite(image_path, img_Slice)
            image = bpy.data.images.load(image_path)
            image.pack()
            # print(f"{img_Name} was processed...")

        #########################################################################################
        # Get slices list :
        MaxSp = max(Vector(Sp))
        if MaxSp < 0.25:
            SampleRatio = round(MaxSp / 0.25, 2)
            Image3D_255 = ResizeImage(sitkImage=Image3D_255, Ratio=SampleRatio)
            DcmInfo["RenderSz"] = Image3D_255.GetSize()
            DcmInfo["RenderSp"] = Image3D_255.GetSpacing()

        Array = sitk.GetArrayFromImage(Image3D_255)
        slices = [np.flipud(Array[i, :, :]) for i in range(Array.shape[0])]
        # slices = [Image3D_255[:, :, i] for i in range(Image3D_255.GetDepth())]

        threads = [
            threading.Thread(
                target=Image3DToPNG,
                args=[i, slices, PngDir, Preffix],
                daemon=True,
            )
            for i in range(len(slices))
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # os.removedirs(PngDir)
        shutil.rmtree(PngDir)
        DcmInfo["CT_Loaded"] = True
        # Set DcmInfo property :
        DcmInfoDict = eval(BDENTAL_4D_Props.DcmInfo)
        DcmInfoDict[Preffix] = DcmInfo
        BDENTAL_4D_Props.DcmInfo = str(DcmInfoDict)
        BDENTAL_4D_Props.UserProjectDir = RelPath(BDENTAL_4D_Props.UserProjectDir)
        bpy.ops.wm.save_mainfile()
        # #################################### debug_04 ####################################
        # debug_04 = Tcounter()
        # message = (
        #     f"PNG images exported (Time : {debug_04-debug_03} secondes)"
        # )
        # print(message)
        # # q.put("PNG images saved...")
        # ##################################################################################

        # #################################### debug_05 ####################################
        # debug_05 = Tcounter()
        # message = f"{Preffix}_CT-SCAN.blend saved (Time = {debug_05-debug_04} secondes)"
        # print(message)
        # q.put("Blender project saved...")
        ##################################################################################

        #############################################################################################
        finish = Tcounter()
        message = f"Data Loaded in {finish-start} secondes"
        print(message)
        # q.put(message)
        #############################################################################################
        message = ["DICOM loaded successfully. "]
        ShowMessageBox(message=message, icon="COLORSET_03_VEC")

        return DcmInfo
    ####### End Load_Dicom_fuction ##############


#######################################################################################
# BDENTAL_4D CT Scan 3DImage File Load :


def Load_3DImage_function(context, q):

    BDENTAL_4D_Props = context.scene.BDENTAL_4D_Props
    UserProjectDir = AbsPath(BDENTAL_4D_Props.UserProjectDir)
    UserImageFile = AbsPath(BDENTAL_4D_Props.UserImageFile)

    #######################################################################################

    if not exists(UserProjectDir):

        message = ["The Selected Project Directory Path is not valid ! "]
        ShowMessageBox(message=message, icon="COLORSET_02_VEC")
        return {"CANCELLED"}

    if not exists(UserImageFile):
        message = [" The Selected Image File Path is not valid ! "]

        ShowMessageBox(message=message, icon="COLORSET_02_VEC")
        return {"CANCELLED"}

    reader = sitk.ImageFileReader()
    IO = reader.GetImageIOFromFileName(UserImageFile)
    FileExt = os.path.splitext(UserImageFile)[1]

    if not IO:
        message = [
            f"{FileExt} files are not Supported! for more info about supported files please refer to Addon wiki "
        ]
        ShowMessageBox(message=message, icon="COLORSET_01_VEC")
        return {"CANCELLED"}

    Image3D = sitk.ReadImage(UserImageFile)
    Depth = Image3D.GetDepth()

    if Depth == 0:
        message = [
            "Can't Build 3D Volume from 2D Image !",
            "for more info about supported files,",
            "please refer to Addon wiki",
        ]
        ShowMessageBox(message=message, icon="COLORSET_01_VEC")
        return {"CANCELLED"}

    ImgFileName = os.path.split(UserImageFile)[1]
    BDENTAL_4D_nrrd = HU_Image = False
    if ImgFileName.startswith("BD") and ImgFileName.endswith("_Image3D255.nrrd"):
        BDENTAL_4D_nrrd = True
    if Image3D.GetPixelIDTypeAsString() in [
        "32-bit signed integer",
        "16-bit signed integer",
    ]:
        HU_Image = True

    if not BDENTAL_4D_nrrd and not HU_Image:
        message = [
            "Only Images with Hunsfield data or BDENTAL_4D nrrd images are supported !"
        ]
        ShowMessageBox(message=message, icon="COLORSET_01_VEC")
        return {"CANCELLED"}
    ###########################################################################################################

    else:

        start = Tcounter()
        ####################################
        # Get Preffix and save file :
        DcmInfoDict = eval(BDENTAL_4D_Props.DcmInfo)
        Preffixs = list(DcmInfoDict.keys())

        for i in range(1, 100):
            Preffix = f"BD{i:03}"
            if not Preffix in Preffixs:
                break
        ########################################################
        Split = split(UserProjectDir)
        ProjectName = Split[-1] or Split[-2]
        BlendFile = f"{ProjectName}_CT-SCAN.blend"
        Blendpath = join(UserProjectDir, BlendFile)

        if not exists(Blendpath) or bpy.context.blend_data.filepath == Blendpath:
            bpy.ops.wm.save_as_mainfile(filepath=Blendpath)
        else:
            bpy.ops.wm.save_mainfile()
        Image3D = sitk.ReadImage(UserImageFile)

        # Start Reading Dicom data :
        ######################################################################################
        # Get Dicom Info :
        reader = sitk.ImageFileReader()
        reader.SetFileName(UserImageFile)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()

        Image3D = reader.Execute()

        Sp = Spacing = Image3D.GetSpacing()
        Sz = Size = Image3D.GetSize()
        Dims = Dimensions = Image3D.GetDimension()
        Origin = Image3D.GetOrigin()
        Direction = Image3D.GetDirection()

        # calculate Informations :
        D = Direction
        O = Origin
        DirectionMatrix_4x4 = Matrix(
            (
                (D[0], D[1], D[2], 0.0),
                (D[3], D[4], D[5], 0.0),
                (D[6], D[7], D[8], 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        )

        TransMatrix_4x4 = Matrix(
            (
                (1.0, 0.0, 0.0, O[0]),
                (0.0, 1.0, 0.0, O[1]),
                (0.0, 0.0, 1.0, O[2]),
                (0.0, 0.0, 0.0, 1.0),
            )
        )

        VtkTransform_4x4 = TransMatrix_4x4 @ DirectionMatrix_4x4
        P0 = Image3D.TransformContinuousIndexToPhysicalPoint((0, 0, 0))
        P_diagonal = Image3D.TransformContinuousIndexToPhysicalPoint(
            (Sz[0] - 1, Sz[1] - 1, Sz[2] - 1)
        )
        VCenter = (Vector(P0) + Vector(P_diagonal)) * 0.5

        C = VCenter

        TransformMatrix = Matrix(
            (
                (D[0], D[1], D[2], C[0]),
                (D[3], D[4], D[5], C[1]),
                (D[6], D[7], D[8], C[2]),
                (0.0, 0.0, 0.0, 1.0),
            )
        )

        # Set DcmInfo :

        DcmInfo = {
            "UserProjectDir": RelPath(UserProjectDir),
            "Preffix": Preffix,
            "RenderSz": Sz,
            "RenderSp": Sp,
            "PixelType": Image3D.GetPixelIDTypeAsString(),
            "Wmin": Wmin,
            "Wmax": Wmax,
            "Size": Sz,
            "Dims": Dims,
            "Spacing": Sp,
            "Origin": Origin,
            "Direction": Direction,
            "TransformMatrix": TransformMatrix,
            "DirectionMatrix_4x4": DirectionMatrix_4x4,
            "TransMatrix_4x4": TransMatrix_4x4,
            "VtkTransform_4x4": VtkTransform_4x4,
            "VolumeCenter": VCenter,
        }

        tags = {
            "StudyDate": "0008|0020",
            "PatientName": "0010|0010",
            "PatientID": "0010|0020",
            "BirthDate": "0010|0030",
            "WinCenter": "0028|1050",
            "WinWidth": "0028|1051",
        }

        for k, tag in tags.items():

            if tag in reader.GetMetaDataKeys():
                v = reader.GetMetaData(tag)

            else:
                v = ""

            DcmInfo[k] = v
            Image3D.SetMetaData(tag, v)

        #######################################################################################
        # Add directories :
        SlicesDir = join(UserProjectDir, "Slices")
        if not exists(SlicesDir):
            os.makedirs(SlicesDir)
        DcmInfo["SlicesDir"] = RelPath(SlicesDir)

        PngDir = join(UserProjectDir, "PNG")
        if not exists(PngDir):
            os.makedirs(PngDir)

        Nrrd255Path = join(UserProjectDir, f"{Preffix}_Image3D255.nrrd")

        DcmInfo["Nrrd255Path"] = RelPath(Nrrd255Path)

        if BDENTAL_4D_nrrd:
            Image3D_255 = Image3D

        else:
            #######################################################################################
            # set IntensityWindowing  :
            Image3D_255 = sitk.Cast(
                sitk.IntensityWindowing(
                    Image3D,
                    windowMinimum=Wmin,
                    windowMaximum=Wmax,
                    outputMinimum=0.0,
                    outputMaximum=255.0,
                ),
                sitk.sitkUInt8,
            )

        # Convert Dicom to nrrd file :
        # sitk.WriteImage(Image3D, NrrdHuPath)
        sitk.WriteImage(Image3D_255, Nrrd255Path)

        #############################################################################################
        # MultiThreading PNG Writer:
        #########################################################################################
        def Image3DToPNG(i, slices, PngDir, Preffix):
            img_Slice = slices[i]
            img_Name = f"{Preffix}_img{i:04}.png"
            image_path = join(PngDir, img_Name)
            cv2.imwrite(image_path, img_Slice)
            image = bpy.data.images.load(image_path)
            image.pack()
            # print(f"{img_Name} was processed...")

        #########################################################################################
        # Get slices list :
        MaxSp = max(Vector(Sp))
        if MaxSp < 0.25:
            SampleRatio = round(MaxSp / 0.25, 2)
            Image3D_255 = ResizeImage(sitkImage=Image3D_255, Ratio=SampleRatio)
            DcmInfo["RenderSz"] = Image3D_255.GetSize()
            DcmInfo["RenderSp"] = Image3D_255.GetSpacing()

        Array = sitk.GetArrayFromImage(Image3D_255)
        slices = [np.flipud(Array[i, :, :]) for i in range(Array.shape[0])]
        # slices = [Image3D_255[:, :, i] for i in range(Image3D_255.GetDepth())]

        threads = [
            threading.Thread(
                target=Image3DToPNG,
                args=[i, slices, PngDir, Preffix],
                daemon=True,
            )
            for i in range(len(slices))
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # os.removedirs(PngDir)
        shutil.rmtree(PngDir)
        DcmInfo["CT_Loaded"] = True

        # Set DcmInfo property :
        DcmInfoDict = eval(BDENTAL_4D_Props.DcmInfo)
        DcmInfoDict[Preffix] = DcmInfo
        BDENTAL_4D_Props.DcmInfo = str(DcmInfoDict)
        BDENTAL_4D_Props.UserProjectDir = RelPath(BDENTAL_4D_Props.UserProjectDir)
        bpy.ops.wm.save_mainfile()

        #############################################################################################
        finish = Tcounter()
        print(f"Data Loaded in {finish-start} second(s)")
        #############################################################################################

        return DcmInfo


##########################################################################################
######################### BDENTAL_4D Volume Render : ########################################
##########################################################################################
class BDENTAL_4D_OT_Volume_Render(bpy.types.Operator):
    """ Volume Render """

    bl_idname = "bdental.volume_render"
    bl_label = "LOAD SCAN"

    q = Queue()

    def execute(self, context):

        Start = Tcounter()
        print("Data Loading START...")

        global ShadersBlendFile
        global GpShader

        BDENTAL_4D_Props = context.scene.BDENTAL_4D_Props

        DataType = BDENTAL_4D_Props.DataType
        if DataType == "DICOM Series":
            DcmInfo = Load_Dicom_funtion(context, self.q)
        if DataType == "3D Image File":
            DcmInfo = Load_3DImage_function(context, self.q)

        UserProjectDir = AbsPath(BDENTAL_4D_Props.UserProjectDir)
        Preffix = DcmInfo["Preffix"]
        Wmin = DcmInfo["Wmin"]
        Wmax = DcmInfo["Wmax"]
        # PngDir = AbsPath(BDENTAL_4D_Props.PngDir)
        print("\n##########################\n")
        print("Voxel Rendering START...")
        VolumeRender(DcmInfo, GpShader, ShadersBlendFile)
        scn = bpy.context.scene
        scn.render.engine = "BLENDER_EEVEE"
        BDENTAL_4D_Props.GroupNodeName = GpShader

        if GpShader == "VGS_Marcos_modified":
            GpNode = bpy.data.node_groups.get(f"{Preffix}_{GpShader}")
            Low_Treshold = GpNode.nodes["Low_Treshold"].outputs[0]
            Low_Treshold.default_value = 600
            WminNode = GpNode.nodes["WminNode"].outputs[0]
            WminNode.default_value = Wmin
            WmaxNode = GpNode.nodes["WmaxNode"].outputs[0]
            WmaxNode.default_value = Wmax

            # newdriver = Low_Treshold.driver_add("default_value")
            # newdriver.driver.type = "AVERAGE"
            # var = newdriver.driver.variables.new()
            # var.name = "Treshold"
            # var.type = "SINGLE_PROP"
            # var.targets[0].id_type = "SCENE"
            # var.targets[0].id = bpy.context.scene
            # var.targets[0].data_path = "BDENTAL_4D_Props.Treshold"
            # newdriver.driver.expression = "Treshold"

        if GpShader == "VGS_Dakir_01":
            # Add Treshold Driver :
            GpNode = bpy.data.node_groups.get(f"{Preffix}_{GpShader}")
            value = (600 - Wmin) / (Wmax - Wmin)
            treshramp = GpNode.nodes["TresholdRamp"].color_ramp.elements[0] = value

            # newdriver = treshramp.driver_add("position")
            # newdriver.driver.type = "SCRIPTED"
            # var = newdriver.driver.variables.new()
            # var.name = "Treshold"
            # var.type = "SINGLE_PROP"
            # var.targets[0].id_type = "SCENE"
            # var.targets[0].id = bpy.context.scene
            # var.targets[0].data_path = "BDENTAL_4D_Props.Treshold"
            # newdriver.driver.expression = f"(Treshold-{Wmin})/{Wmax-Wmin}"

        BDENTAL_4D_Props.CT_Rendered = True
        bpy.ops.view3d.view_selected(use_all_regions=False)
        bpy.ops.wm.save_mainfile()

        # post_handlers = bpy.app.handlers.depsgraph_update_post
        # [
        #     post_handlers.remove(h)
        #     for h in post_handlers
        #     if h.__name__ == "BDENTAL_4D_TresholdUpdate"
        # ]
        # post_handlers.append(BDENTAL_4D_TresholdUpdate)

        # bpy.ops.wm.save_mainfile()

        Finish = Tcounter()

        print(f"Finished (Time : {Finish-Start}")

        return {"FINISHED"}


class BDENTAL_4D_OT_TresholdUpdate(bpy.types.Operator):
    """ Add treshold Update Handler  """

    bl_idname = "bdental.tresholdupdate"
    bl_label = "Update Treshold"

    def execute(self, context):
        post_handlers = bpy.app.handlers.depsgraph_update_post
        [
            post_handlers.remove(h)
            for h in post_handlers
            if h.__name__ == "BDENTAL_4D_TresholdUpdate"
        ]
        post_handlers.append(BDENTAL_4D_TresholdUpdate)

        return {"FINISHED"}


##########################################################################################
######################### BDENTAL_4D Add Slices : ########################################
##########################################################################################


class BDENTAL_4D_OT_AddSlices(bpy.types.Operator):
    """ Add Volume Slices """

    bl_idname = "bdental.addslices"
    bl_label = "SLICE VOLUME"

    def execute(self, context):
        BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props

        Active_Obj = bpy.context.view_layer.objects.active

        if not Active_Obj:
            message = [" Please select CTVOLUME or SEGMENTATION ! "]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")
            return {"CANCELLED"}
        else:
            Conditions = [
                not Active_Obj.name.startswith("BD"),
                not Active_Obj.name.endswith(("_CTVolume", "SEGMENTATION")),
                Active_Obj.select_get() == False,
            ]
            if Conditions[0] or Conditions[1] or Conditions[2]:
                message = [" Please select CTVOLUME or SEGMENTATION ! "]
                ShowMessageBox(message=message, icon="COLORSET_02_VEC")
                return {"CANCELLED"}
            else:
                Vol = Active_Obj
                Preffix = Vol.name[:5]
                DcmInfoDict = eval(BDENTAL_4D_Props.DcmInfo)
                DcmInfo = DcmInfoDict[Preffix]

                # SLICES_Coll = bpy.context.scene.collection.children.get('SLICES')
                # if SLICES_Coll :
                #     SLICES_Coll.hide_viewport = False

                AxialPlane = AddAxialSlice(Preffix, DcmInfo)
                MoveToCollection(obj=AxialPlane, CollName="SLICES")

                CoronalPlane = AddCoronalSlice(Preffix, DcmInfo)
                MoveToCollection(obj=CoronalPlane, CollName="SLICES")

                SagitalPlane = AddSagitalSlice(Preffix, DcmInfo)
                MoveToCollection(obj=SagitalPlane, CollName="SLICES")

                # Add Cameras :

                bpy.context.scene.render.resolution_x = 512
                bpy.context.scene.render.resolution_y = 512

                [
                    bpy.data.cameras.remove(cam)
                    for cam in bpy.data.cameras
                    if f"{AxialPlane.name}_CAM" in cam.name
                ]
                AxialCam = Add_Cam_To_Plane(AxialPlane, CamDistance=100, ClipOffset=1)
                MoveToCollection(obj=AxialCam, CollName="SLICES-CAMERAS")

                [
                    bpy.data.cameras.remove(cam)
                    for cam in bpy.data.cameras
                    if f"{CoronalPlane.name}_CAM" in cam.name
                ]
                CoronalCam = Add_Cam_To_Plane(
                    CoronalPlane, CamDistance=100, ClipOffset=1
                )
                MoveToCollection(obj=CoronalCam, CollName="SLICES-CAMERAS")

                [
                    bpy.data.cameras.remove(cam)
                    for cam in bpy.data.cameras
                    if f"{SagitalPlane.name}_CAM" in cam.name
                ]
                SagitalCam = Add_Cam_To_Plane(
                    SagitalPlane, CamDistance=100, ClipOffset=1
                )
                MoveToCollection(obj=SagitalCam, CollName="SLICES-CAMERAS")

                for obj in bpy.data.objects:
                    if obj.name == f"{Preffix}_SLICES_POINTER":
                        bpy.data.objects.remove(obj)

                bpy.ops.object.empty_add(
                    type="PLAIN_AXES",
                    align="WORLD",
                    location=AxialPlane.location,
                    scale=(1, 1, 1),
                )
                SLICES_POINTER = bpy.context.object
                SLICES_POINTER.empty_display_size = 20
                SLICES_POINTER.show_name = True
                SLICES_POINTER.show_in_front = True
                SLICES_POINTER.name = f"{Preffix}_SLICES_POINTER"

                Override, _, _ = CtxOverride(bpy.context)

                bpy.ops.object.select_all(Override, action="DESELECT")
                AxialPlane.select_set(True)
                CoronalPlane.select_set(True)
                SagitalPlane.select_set(True)
                SLICES_POINTER.select_set(True)
                bpy.context.view_layer.objects.active = SLICES_POINTER
                bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)
                bpy.ops.object.select_all(Override, action="DESELECT")
                SLICES_POINTER.select_set(True)
                Vol.select_set(True)
                bpy.context.view_layer.objects.active = Vol
                bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)

                bpy.ops.object.select_all(Override, action="DESELECT")
                SLICES_POINTER.select_set(True)
                bpy.context.view_layer.objects.active = SLICES_POINTER
                MoveToCollection(obj=SLICES_POINTER, CollName="SLICES_POINTERS")

                return {"FINISHED"}


###############################################################################
####################### BDENTAL_4D_FULL VOLUME to Mesh : ################################
##############################################################################
class BDENTAL_4D_OT_MultiTreshSegment(bpy.types.Operator):
    """ Add a mesh Segmentation using Treshold """

    bl_idname = "bdental.multitresh_segment"
    bl_label = "SEGMENTATION"

    TimingDict = {}

    def ImportMeshStl(self, Segment, SegmentStlPath, SegmentColor):

        # import stl to blender scene :
        bpy.ops.import_mesh.stl(filepath=SegmentStlPath)
        obj = bpy.context.object
        obj.name = f"{self.Preffix}_{Segment}_SEGMENTATION"
        obj.data.name = f"{self.Preffix}_{Segment}_mesh"

        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")

        self.step7 = Tcounter()
        self.TimingDict["Mesh Import"] = self.step7 - self.step6

        ############### step 8 : Add material... #########################
        mat = bpy.data.materials.get(obj.name) or bpy.data.materials.new(obj.name)
        mat.diffuse_color = SegmentColor
        obj.data.materials.append(mat)
        MoveToCollection(obj=obj, CollName="SEGMENTS")
        bpy.ops.object.shade_smooth()

        bpy.ops.object.modifier_add(type="CORRECTIVE_SMOOTH")
        bpy.context.object.modifiers["CorrectiveSmooth"].iterations = 2
        bpy.context.object.modifiers["CorrectiveSmooth"].use_only_smooth = True
        bpy.ops.object.modifier_apply(modifier="CorrectiveSmooth")

        self.step8 = Tcounter()
        self.TimingDict["Add material"] = self.step8 - self.step7
        print(f"{Segment} Mesh Import Finished")

        return obj

        # self.q.put(["End"])

    def DicomToStl(self, Segment, Image3D):
        print(f"{Segment} processing ...")
        # Load Infos :
        #########################################################################
        BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props
        UserProjectDir = AbsPath(BDENTAL_4D_Props.UserProjectDir)
        DcmInfo = self.DcmInfo
        Origin = DcmInfo["Origin"]
        VtkTransform_4x4 = DcmInfo["VtkTransform_4x4"]
        TransformMatrix = DcmInfo["TransformMatrix"]
        VtkMatrix_4x4 = (
            self.Vol.matrix_world @ TransformMatrix.inverted() @ VtkTransform_4x4
        )

        VtkMatrix = list(np.array(VtkMatrix_4x4).ravel())

        SmoothIterations = SmthIter = 5
        Thikness = 1

        SegmentTreshold = self.SegmentsDict[Segment]["Treshold"]
        SegmentColor = self.SegmentsDict[Segment]["Color"]
        SegmentStlPath = join(UserProjectDir, f"{Segment}_SEGMENTATION.stl")

        # Convert Hu treshold value to 0-255 UINT8 :
        Treshold255 = HuTo255(Hu=SegmentTreshold, Wmin=Wmin, Wmax=Wmax)
        if Treshold255 == 0:
            Treshold255 = 1
        elif Treshold255 == 255:
            Treshold255 = 254

        ############### step 2 : Extracting mesh... #########################
        # print("Extracting mesh...")
        vtkImage = sitkTovtk(sitkImage=Image3D)

        ExtractedMesh = vtk_MC_Func(vtkImage=vtkImage, Treshold=Treshold255)
        Mesh = ExtractedMesh

        polysCount = Mesh.GetNumberOfPolys()
        polysLimit = 800000

        self.step2 = Tcounter()
        self.TimingDict["Mesh Extraction Time"] = self.step2 - self.step1
        print(f"{Segment} Mesh Extraction Finished")
        ############### step 3 : mesh Reduction... #########################
        if polysCount > polysLimit:

            Reduction = round(1 - (polysLimit / polysCount), 2)
            ReductedMesh = vtkMeshReduction(
                q=self.q,
                mesh=Mesh,
                reduction=Reduction,
                step="Mesh Reduction",
                start=0.11,
                finish=0.75,
            )
            Mesh = ReductedMesh

        self.step3 = Tcounter()
        self.TimingDict["Mesh Reduction Time"] = self.step3 - self.step2
        print(f"{Segment} Mesh Reduction Finished")
        ############### step 4 : mesh Smoothing... #########################
        SmoothedMesh = vtkSmoothMesh(
            q=self.q,
            mesh=Mesh,
            Iterations=SmthIter,
            step="Mesh Smoothing",
            start=0.76,
            finish=0.78,
        )

        self.step4 = Tcounter()
        self.TimingDict["Mesh Smoothing Time"] = self.step4 - self.step3
        print(f"{Segment} Mesh Smoothing Finished")
        ############### step 5 : Set mesh orientation... #########################
        TransformedMesh = vtkTransformMesh(
            mesh=SmoothedMesh,
            Matrix=VtkMatrix,
        )
        self.step5 = Tcounter()
        self.TimingDict["Mesh Orientation"] = self.step5 - self.step4
        print(f"{Segment} Mesh Orientation Finished")
        ############### step 6 : exporting mesh stl... #########################
        writer = vtk.vtkSTLWriter()
        writer.SetInputData(TransformedMesh)
        writer.SetFileTypeToBinary()
        writer.SetFileName(SegmentStlPath)
        writer.Write()

        self.step6 = Tcounter()
        self.TimingDict["Mesh Export"] = self.step6 - self.step5
        print(f"{Segment} Mesh Export Finished")
        self.Exported.put([Segment, SegmentStlPath, SegmentColor])

    def execute(self, context):

        self.counter_start = Tcounter()

        BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props
        Active_Obj = bpy.context.view_layer.objects.active

        if not Active_Obj:
            message = [" Please select CTVOLUME for segmentation ! "]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")
            return {"CANCELLED"}
        else:
            Conditions = [
                not Active_Obj.name.startswith("BD"),
                not Active_Obj.name.endswith("_CTVolume"),
                Active_Obj.select_get() == False,
            ]

            if Conditions[0] or Conditions[1] or Conditions[2]:
                message = [" Please select CTVOLUME for segmentation ! "]
                ShowMessageBox(message=message, icon="COLORSET_02_VEC")
                return {"CANCELLED"}

            else:

                self.Soft = BDENTAL_4D_Props.SoftBool
                self.Bone = BDENTAL_4D_Props.BoneBool
                self.Teeth = BDENTAL_4D_Props.TeethBool

                self.SoftTresh = BDENTAL_4D_Props.SoftTreshold
                self.BoneTresh = BDENTAL_4D_Props.BoneTreshold
                self.TeethTresh = BDENTAL_4D_Props.TeethTreshold

                self.SoftSegmentColor = BDENTAL_4D_Props.SoftSegmentColor
                self.BoneSegmentColor = BDENTAL_4D_Props.BoneSegmentColor
                self.TeethSegmentColor = BDENTAL_4D_Props.TeethSegmentColor

                self.SegmentsDict = {
                    "Soft": {
                        "State": self.Soft,
                        "Treshold": self.SoftTresh,
                        "Color": self.SoftSegmentColor,
                    },
                    "Bone": {
                        "State": self.Bone,
                        "Treshold": self.BoneTresh,
                        "Color": self.BoneSegmentColor,
                    },
                    "Teeth": {
                        "State": self.Teeth,
                        "Treshold": self.TeethTresh,
                        "Color": self.TeethSegmentColor,
                    },
                }

                ActiveSegmentsList = [
                    k for k, v in self.SegmentsDict.items() if v["State"]
                ]

                if not ActiveSegmentsList:
                    message = [
                        " Please check at least 1 segmentation ! ",
                        "(Soft - Bone - Teeth)",
                    ]
                    ShowMessageBox(message=message, icon="COLORSET_02_VEC")
                    return {"CANCELLED"}
                else:

                    self.Vol = Active_Obj
                    self.Preffix = self.Vol.name[:5]
                    DcmInfoDict = eval(BDENTAL_4D_Props.DcmInfo)
                    self.DcmInfo = DcmInfoDict[self.Preffix]
                    self.Nrrd255Path = AbsPath(self.DcmInfo["Nrrd255Path"])
                    self.q = Queue()
                    self.Exported = Queue()

                    if not exists(self.Nrrd255Path):

                        message = [" Image File not Found in Project Folder ! "]
                        ShowMessageBox(message=message, icon="COLORSET_01_VEC")
                        return {"CANCELLED"}

                    else:

                        ############### step 1 : Reading DICOM #########################
                        self.step1 = Tcounter()
                        self.TimingDict["Read DICOM"] = self.step1 - self.counter_start
                        print(f"step 1 : Read DICOM ({self.step1-self.counter_start})")

                        Image3D = sitk.ReadImage(self.Nrrd255Path)
                        # Sz = Image3D.GetSize()
                        Sp = Image3D.GetSpacing()
                        MaxSp = max(Vector(Sp))
                        if MaxSp < 0.3:
                            SampleRatio = round(MaxSp / 0.3, 2)
                            ResizedImage = ResizeImage(
                                sitkImage=Image3D, Ratio=SampleRatio
                            )
                            Image3D = ResizedImage
                            print(f"Image DOWN Sampled : SampleRatio = {SampleRatio}")

                        ############### step 2 : Dicom To Stl Threads #########################

                        self.MeshesCount = len(ActiveSegmentsList)
                        Imported_Meshes = []
                        Threads = [
                            threading.Thread(
                                target=self.DicomToStl,
                                args=[Segment, Image3D],
                                daemon=True,
                            )
                            for Segment in ActiveSegmentsList
                        ]
                        for t in Threads:
                            t.start()
                        count = 0
                        while count < self.MeshesCount:
                            if not self.Exported.empty():
                                (
                                    Segment,
                                    SegmentStlPath,
                                    SegmentColor,
                                ) = self.Exported.get()
                                for i in range(10):
                                    if not exists(SegmentStlPath):
                                        sleep(0.1)
                                    else:
                                        break
                                obj = self.ImportMeshStl(
                                    Segment, SegmentStlPath, SegmentColor
                                )
                                Imported_Meshes.append(obj)
                                count += 1
                            else:
                                sleep(0.1)
                        for t in Threads:
                            t.join()

                        for obj in Imported_Meshes:
                            bpy.ops.object.select_all(action="DESELECT")
                            obj.select_set(True)
                            bpy.context.view_layer.objects.active = obj
                            for i in range(3):
                                obj.lock_location[i] = True
                                obj.lock_rotation[i] = True
                                obj.lock_scale[i] = True

                        bpy.ops.object.select_all(action="DESELECT")
                        for obj in Imported_Meshes:
                            obj.select_set(True)
                        self.Vol.select_set(True)
                        bpy.context.view_layer.objects.active = self.Vol
                        bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)
                        bpy.ops.object.select_all(action="DESELECT")

                        self.counter_finish = Tcounter()
                        self.TimingDict["Total Time"] = (
                            self.counter_finish - self.counter_start
                        )

                        print(self.TimingDict)

                        return {"FINISHED"}


###############################################################################
####################### BDENTAL_4D VOLUME to Mesh : ################################
##############################################################################
# class BDENTAL_4D_OT_TreshSegment(bpy.types.Operator):
#     """ Add a mesh Segmentation using Treshold """

#     bl_idname = "bdental.tresh_segment"
#     bl_label = "SEGMENTATION"

#     SegmentName: StringProperty(
#         name="Segmentation Name",
#         default="TEST",
#         description="Segmentation Name",
#     )
#     SegmentColor: FloatVectorProperty(
#         name="Segmentation Color",
#         description="Segmentation Color",
#         default=[0.44, 0.4, 0.5, 1.0],  # (0.8, 0.46, 0.4, 1.0),
#         soft_min=0.0,
#         soft_max=1.0,
#         size=4,
#         subtype="COLOR",
#     )

#     TimingDict = {}

#     def invoke(self, context, event):

#         BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props

#         Active_Obj = bpy.context.view_layer.objects.active

#         if not Active_Obj:
#             message = [" Please select CTVOLUME for segmentation ! "]
#             ShowMessageBox(message=message, icon="COLORSET_02_VEC")
#             return {"CANCELLED"}
#         else:
#             Conditions = [
#                 not Active_Obj.name.startswith("BD"),
#                 not Active_Obj.name.endswith("_CTVolume"),
#                 Active_Obj.select_get() == False,
#             ]

#             if Conditions[0] or Conditions[1] or Conditions[2]:
#                 message = [" Please select CTVOLUME for segmentation ! "]
#                 ShowMessageBox(message=message, icon="COLORSET_02_VEC")
#                 return {"CANCELLED"}

#             else:
#                 self.Vol = Active_Obj
#                 self.Preffix = self.Vol.name[:5]
#                 DcmInfoDict = eval(BDENTAL_4D_Props.DcmInfo)
#                 self.DcmInfo = DcmInfoDict[self.Preffix]
#                 self.Nrrd255Path = AbsPath(self.DcmInfo["Nrrd255Path"])
#                 self.Treshold = BDENTAL_4D_Props.Treshold

#                 if exists(self.Nrrd255Path):
#                     if GpShader == "VGS_Marcos_modified":
#                         GpNode = bpy.data.node_groups.get(f"{self.Preffix}_{GpShader}")
#                         ColorPresetRamp = GpNode.nodes["ColorPresetRamp"].color_ramp
#                         value = (self.Treshold - Wmin) / (Wmax - Wmin)
#                         TreshColor = [
#                             round(c, 2) for c in ColorPresetRamp.evaluate(value)[0:3]
#                         ]
#                         self.SegmentColor = TreshColor + [1.0]
#                     self.q = Queue()
#                     wm = context.window_manager
#                     return wm.invoke_props_dialog(self)

#                 else:
#                     message = [" Image File not Found in Project Folder ! "]
#                     ShowMessageBox(message=message, icon="COLORSET_01_VEC")
#                     return {"CANCELLED"}

#     def DicomToMesh(self):
#         counter_start = Tcounter()

#         self.q.put(["GuessTime", "PROGRESS : Extracting mesh...", "", 0.0, 0.1, 2])
#         # Load Infos :
#         #########################################################################
#         BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props
#         # NrrdHuPath = BDENTAL_4D_Props.NrrdHuPath
#         Nrrd255Path = self.Nrrd255Path
#         UserProjectDir = AbsPath(BDENTAL_4D_Props.UserProjectDir)
#         DcmInfo = self.DcmInfo
#         Origin = DcmInfo["Origin"]
#         VtkTransform_4x4 = DcmInfo["VtkTransform_4x4"]
#         VtkMatrix = list(np.array(VtkTransform_4x4).ravel())
#         Treshold = self.Treshold

#         StlPath = join(UserProjectDir, f"{self.SegmentName}_SEGMENTATION.stl")
#         Thikness = 1
#         # Reduction = 0.9
#         SmoothIterations = SmthIter = 5

#         ############### step 1 : Reading DICOM #########################
#         # self.q.put(["GuessTime", "PROGRESS : Reading DICOM...", "", 0, 0.1, 1])

#         Image3D = sitk.ReadImage(Nrrd255Path)
#         Sz = Image3D.GetSize()
#         Sp = Image3D.GetSpacing()
#         MaxSp = max(Vector(Sp))
#         if MaxSp < 0.3:
#             SampleRatio = round(MaxSp / 0.3, 2)
#             ResizedImage = ResizeImage(sitkImage=Image3D, Ratio=SampleRatio)
#             Image3D = ResizedImage
#             # print(f"Image DOWN Sampled : SampleRatio = {SampleRatio}")

#         # Convert Hu treshold value to 0-255 UINT8 :
#         Treshold255 = HuTo255(Hu=Treshold, Wmin=Wmin, Wmax=Wmax)
#         if Treshold255 == 0:
#             Treshold255 = 1
#         elif Treshold255 == 255:
#             Treshold255 = 254

#         step1 = Tcounter()
#         self.TimingDict["Read DICOM"] = step1 - counter_start
#         # print(f"step 1 : Read DICOM ({step1-start})")

#         ############### step 2 : Extracting mesh... #########################
#         # self.q.put(["GuessTime", "PROGRESS : Extracting mesh...", "", 0.0, 0.1, 2])

#         # print("Extracting mesh...")
#         vtkImage = sitkTovtk(sitkImage=Image3D)

#         ExtractedMesh = vtk_MC_Func(vtkImage=vtkImage, Treshold=Treshold255)
#         Mesh = ExtractedMesh

#         polysCount = Mesh.GetNumberOfPolys()
#         polysLimit = 800000

#         # step1 = Tcounter()
#         # print(f"before reduction finished in : {step1-start} secondes")
#         step2 = Tcounter()
#         self.TimingDict["extract mesh"] = step2 - step1
#         # print(f"step 2 : extract mesh ({step2-step1})")

#         ############### step 3 : mesh Reduction... #########################
#         if polysCount > polysLimit:
#             # print(f"Hight polygons count, : ({polysCount}) Mesh will be reduced...")
#             Reduction = round(1 - (polysLimit / polysCount), 2)
#             # print(f"MESH REDUCTION: Ratio = ({Reduction}) ...")

#             ReductedMesh = vtkMeshReduction(
#                 q=self.q,
#                 mesh=Mesh,
#                 reduction=Reduction,
#                 step="Mesh Reduction",
#                 start=0.11,
#                 finish=0.75,
#             )
#             Mesh = ReductedMesh
#             # print(f"Reduced Mesh polygons count : {Mesh.GetNumberOfPolys()} ...")
#             # step2 = Tcounter()
#             # print(f"reduction finished in : {step2-step1} secondes")
#         # else:
#         # print(f"Original mesh polygons count is Optimal : ({polysCount})...")
#         step3 = Tcounter()
#         self.TimingDict["Reduct mesh"] = step3 - step2
#         # print(f"step 3 : Reduct mesh ({step3-step2})")

#         ############### step 4 : mesh Smoothing... #########################
#         # print("SMOOTHING...")
#         SmoothedMesh = vtkSmoothMesh(
#             q=self.q,
#             mesh=Mesh,
#             Iterations=SmthIter,
#             step="Mesh Orientation",
#             start=0.76,
#             finish=0.78,
#         )
#         step3 = Tcounter()
#         # try:
#         #     print(f"SMOOTHING finished in : {step3-step2} secondes...")
#         # except Exception:
#         #     print(f"SMOOTHING finished in : {step3-step1} secondes (no Reduction!)...")
#         step4 = Tcounter()
#         self.TimingDict["Smooth mesh"] = step4 - step3
#         # print(f"step 4 : Smooth mesh ({step4-step3})")

#         ############### step 5 : Set mesh orientation... #########################
#         # print("SET MESH ORIENTATION...")
#         TransformedMesh = vtkTransformMesh(
#             mesh=SmoothedMesh,
#             Matrix=VtkMatrix,
#         )
#         step5 = Tcounter()
#         self.TimingDict["Mesh Transform"] = step5 - step4
#         # print(f"step 5 : set mesh orientation({step5-step4})")

#         ############### step 6 : exporting mesh stl... #########################
#         self.q.put(
#             [
#                 "GuessTime",
#                 "PROGRESS : exporting mesh stl...",
#                 "",
#                 0.79,
#                 0.83,
#                 2,
#             ]
#         )

#         # print("WRITING...")
#         writer = vtk.vtkSTLWriter()
#         writer.SetInputData(TransformedMesh)
#         writer.SetFileTypeToBinary()
#         writer.SetFileName(StlPath)
#         writer.Write()

#         # step4 = Tcounter()
#         # print(f"WRITING finished in : {step4-step3} secondes")
#         step6 = Tcounter()
#         self.TimingDict["Export mesh"] = step6 - step5
#         # print(f"step 6 : Export mesh ({step6-step5})")

#         ############### step 7 : Importing mesh to Blender... #########################
#         self.q.put(["GuessTime", "PROGRESS : Importing mesh...", "", 0.84, 0.97, 8])

#         # print("IMPORTING...")
#         # import stl to blender scene :
#         bpy.ops.import_mesh.stl(filepath=StlPath)
#         obj = bpy.context.object
#         obj.name = f"{self.Preffix}_{self.SegmentName}_SEGMENTATION"
#         obj.data.name = f"{self.Preffix}_{self.SegmentName}_mesh"

#         bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")

#         step7 = Tcounter()
#         self.TimingDict["Import mesh"] = step7 - step6
#         # print(f"step 7 : Import mesh({step7-step6})")
#         ############### step 8 : Add material... #########################
#         self.q.put(["GuessTime", "PROGRESS : Add material...", "", 0.98, 0.99, 1])

#         # print("ADD COLOR MATERIAL")
#         mat = bpy.data.materials.get(obj.name) or bpy.data.materials.new(obj.name)
#         mat.diffuse_color = self.SegmentColor
#         obj.data.materials.append(mat)
#         MoveToCollection(obj=obj, CollName="SEGMENTS")
#         bpy.ops.object.shade_smooth()

#         bpy.ops.object.modifier_add(type="CORRECTIVE_SMOOTH")
#         bpy.context.object.modifiers["CorrectiveSmooth"].iterations = 3
#         bpy.context.object.modifiers["CorrectiveSmooth"].use_only_smooth = True

#         # step5 = Tcounter()
#         # print(f"Blender importing finished in : {step5-step4} secondes")

#         step8 = Tcounter()
#         self.TimingDict["Add material"] = step8 - step7
#         # print(f"step 8 : Add material({step8-step7})")

#         self.q.put(["End"])
#         counter_finish = Tcounter()
#         self.TimingDict["Total Time"] = counter_finish - counter_start

#     def execute(self, context):
#         counter_start = Tcounter()
#         # TerminalProgressBar = BDENTAL_4D_Utils.TerminalProgressBar
#         CV2_progress_bar = BDENTAL_4D_Utils.CV2_progress_bar
#         # t1 = threading.Thread(
#         #     target=TerminalProgressBar, args=[self.q, counter_start], daemon=True
#         # )
#         t2 = threading.Thread(target=CV2_progress_bar, args=[self.q], daemon=True)

#         # t1.start()
#         t2.start()
#         self.DicomToMesh()
#         # t1.join()
#         t2.join()

#         # self.DicomToMesh()
#         # t1.join()

#         # print("\n")
#         # print(self.TimingDict)

#         return {"FINISHED"}


class BDENTAL_4D_OT_MultiView(bpy.types.Operator):
    """ MultiView Toggle """

    bl_idname = "bdental.multiview"
    bl_label = "MULTI-VIEW"

    def execute(self, context):

        BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props

        Active_Obj = bpy.context.view_layer.objects.active

        if not Active_Obj:
            message = [" Please select CTVOLUME or SEGMENTATION ! "]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")
            return {"CANCELLED"}
        else:
            Conditions = [
                not Active_Obj.name.startswith("BD"),
                not Active_Obj.name.endswith(
                    ("_CTVolume", "SEGMENTATION", "_SLICES_POINTER")
                ),
                Active_Obj.select_get() == False,
            ]
            if Conditions[0] or Conditions[1] or Conditions[2]:
                message = [
                    " Please select CTVOLUME or SEGMENTATION or _SLICES_POINTER ! "
                ]
                ShowMessageBox(message=message, icon="COLORSET_02_VEC")
                return {"CANCELLED"}
            else:
                Preffix = Active_Obj.name[:5]
                AxialPlane = bpy.data.objects.get(f"1_{Preffix}_AXIAL_SLICE")
                CoronalPlane = bpy.data.objects.get(f"2_{Preffix}_CORONAL_SLICE")
                SagitalPlane = bpy.data.objects.get(f"3_{Preffix}_SAGITAL_SLICE")
                SLICES_POINTER = bpy.data.objects.get(f"{Preffix}_SLICES_POINTER")

                if not AxialPlane or not CoronalPlane or not SagitalPlane:
                    message = [
                        "To Add Multi-View Window :",
                        "1 - Please select CTVOLUME or SEGMENTATION",
                        "2 - Click on < SLICE VOLUME > button",
                        "AXIAL, CORONAL and SAGITAL slices will be added",
                        "3 - Click <MULTI-VIEW> button",
                    ]
                    ShowMessageBox(message=message, icon="COLORSET_02_VEC")
                    return {"CANCELLED"}

                else:

                    bpy.context.scene.unit_settings.scale_length = 0.001
                    bpy.context.scene.unit_settings.length_unit = "MILLIMETERS"

                    (
                        MultiView_Window,
                        OUTLINER,
                        PROPERTIES,
                        AXIAL,
                        CORONAL,
                        SAGITAL,
                        VIEW_3D,
                    ) = BDENTAL_4D_MultiView_Toggle(Preffix)
                    MultiView_Screen = MultiView_Window.screen
                    AXIAL_Space3D = [
                        Space for Space in AXIAL.spaces if Space.type == "VIEW_3D"
                    ][0]
                    AXIAL_Region = [
                        reg for reg in AXIAL.regions if reg.type == "WINDOW"
                    ][0]

                    CORONAL_Space3D = [
                        Space for Space in CORONAL.spaces if Space.type == "VIEW_3D"
                    ][0]
                    CORONAL_Region = [
                        reg for reg in CORONAL.regions if reg.type == "WINDOW"
                    ][0]

                    SAGITAL_Space3D = [
                        Space for Space in SAGITAL.spaces if Space.type == "VIEW_3D"
                    ][0]
                    SAGITAL_Region = [
                        reg for reg in SAGITAL.regions if reg.type == "WINDOW"
                    ][0]
                    # AXIAL Cam view toggle :

                    AxialCam = bpy.data.objects.get(f"{AxialPlane.name}_CAM")
                    AXIAL_Space3D.use_local_collections = True
                    AXIAL_Space3D.use_local_camera = True
                    AXIAL_Space3D.camera = AxialCam
                    Override = {
                        "window": MultiView_Window,
                        "screen": MultiView_Screen,
                        "area": AXIAL,
                        "space_data": AXIAL_Space3D,
                        "region": AXIAL_Region,
                    }
                    bpy.ops.view3d.view_camera(Override)

                    # CORONAL Cam view toggle :
                    CoronalCam = bpy.data.objects.get(f"{CoronalPlane.name}_CAM")
                    CORONAL_Space3D.use_local_collections = True
                    CORONAL_Space3D.use_local_camera = True
                    CORONAL_Space3D.camera = CoronalCam
                    Override = {
                        "window": MultiView_Window,
                        "screen": MultiView_Screen,
                        "area": CORONAL,
                        "space_data": CORONAL_Space3D,
                        "region": CORONAL_Region,
                    }
                    bpy.ops.view3d.view_camera(Override)

                    # AXIAL Cam view toggle :
                    SagitalCam = bpy.data.objects.get(f"{SagitalPlane.name}_CAM")
                    SAGITAL_Space3D.use_local_collections = True
                    SAGITAL_Space3D.use_local_camera = True
                    SAGITAL_Space3D.camera = SagitalCam
                    Override = {
                        "window": MultiView_Window,
                        "screen": MultiView_Screen,
                        "area": SAGITAL,
                        "space_data": SAGITAL_Space3D,
                        "region": SAGITAL_Region,
                    }
                    bpy.ops.view3d.view_camera(Override)

                    bpy.ops.object.select_all(Override, action="DESELECT")
                    SLICES_POINTER.select_set(True)
                    bpy.context.view_layer.objects.active = SLICES_POINTER

        return {"FINISHED"}


#######################################################################################
########################### Measurements : Operators ##############################
#######################################################################################
# if event.type == "R":
#             # Add Right Or Point :
#             if event.value == ("PRESS"):
#                 color = (1, 0, 0, 1)  # red
#                 CollName = self.CollName
#                 name = "R_Or"
#                 OldPoint = bpy.data.objects.get(name)
#                 if OldPoint:
#                     bpy.data.objects.remove(OldPoint)
#                 NewPoint = AddMarkupPoint(name, color, CollName)
#                 self.R_Or = NewPoint
#                 bpy.ops.object.select_all(action="DESELECT")
#                 self.Points = [
#                     obj
#                     for obj in bpy.context.scene.objects
#                     if obj.name in self.PointsNames and not obj is self.R_Or
#                 ]
#                 self.Points.append(self.R_Or)

#         if event.type == "L":
#             # Add Left Or point :
#             if event.value == ("PRESS"):
#                 color = (1, 0, 0, 1)  # red
#                 CollName = self.CollName
#                 name = "L_Or"
#                 OldPoint = bpy.data.objects.get(name)
#                 if OldPoint:
#                     bpy.data.objects.remove(OldPoint)
#                 NewPoint = AddMarkupPoint(name, color, CollName)
#                 self.L_Or = NewPoint
#                 bpy.ops.object.select_all(action="DESELECT")
#                 self.Points = [
#                     obj
#                     for obj in bpy.context.scene.objects
#                     if obj.name in self.PointsNames and not obj is self.L_Or
#                 ]
#                 self.Points.append(self.L_Or)

#         if event.shift and event.type == "R":
#             # Add Right Po point :
#             if event.value == ("PRESS"):
#                 color = (1, 0, 0, 1)  # red
#                 CollName = self.CollName
#                 name = "R_Po"
#                 OldPoint = bpy.data.objects.get(name)
#                 if OldPoint:
#                     bpy.data.objects.remove(OldPoint)
#                 NewPoint = AddMarkupPoint(name, color, CollName)
#                 self.R_Po = NewPoint
#                 bpy.ops.object.select_all(action="DESELECT")
#                 self.Points = [
#                     obj
#                     for obj in bpy.context.scene.objects
#                     if obj.name in self.PointsNames and not obj is self.R_Po
#                 ]
#                 self.Points.append(self.R_Po)

#         if event.shift and event.type == "L":
#             # Add Left Po point :
#             if event.value == ("PRESS"):
#                 color = (1, 0, 0, 1)  # red
#                 CollName = self.CollName
#                 name = "L_Po"
#                 OldPoint = bpy.data.objects.get(name)
#                 if OldPoint:
#                     bpy.data.objects.remove(OldPoint)
#                 NewPoint = AddMarkupPoint(name, color, CollName)
#                 self.L_Po = NewPoint
#                 bpy.ops.object.select_all(action="DESELECT")
#                 self.Points = [
#                     obj
#                     for obj in bpy.context.scene.objects
#                     if obj.name in self.PointsNames and not obj is self.L_Po
#                 ]
#                 self.Points.append(self.L_Po)


# def AddFrankfortPoint(PointsList, color, CollName):
#     FrankfortPointsNames = ["R_Or", "L_Or", "R_Po", "L_Po"]
#     if not PointsList:
#         P = AddMarkupPoint(FrankfortPointsNames[0], color, CollName)
#         return P
#     if PointsList:
#         CurrentPointsNames = [P.name for P in PointsList]
#         P_Names = [P for P in FrankfortPointsNames if not P in CurrentPointsNames]
#         if P_Names:
#             P = AddMarkupPoint(P_Names[0], color, CollName)
#             return P
#     else:
#         return None


class BDENTAL_4D_OT_AddReferencePlanes(bpy.types.Operator):
    """ Add Reference Planes"""

    bl_idname = "bdental.add_reference_planes"
    bl_label = "Add REFERENCE PLANES"
    bl_options = {"REGISTER", "UNDO"}

    def modal(self, context, event):

        if (
            event.type
            in [
                "LEFTMOUSE",
                "RIGHTMOUSE",
                "MIDDLEMOUSE",
                "WHEELUPMOUSE",
                "WHEELDOWNMOUSE",
                "N",
                "NUMPAD_2",
                "NUMPAD_4",
                "NUMPAD_6",
                "NUMPAD_8",
                "NUMPAD_1",
                "NUMPAD_3",
                "NUMPAD_5",
                "NUMPAD_7",
                "NUMPAD_9",
            ]
            and event.value == "PRESS"
        ):

            return {"PASS_THROUGH"}
        #########################################
        elif event.type == "RET":
            if event.value == ("PRESS"):

                CurrentPointsNames = [P.name for P in self.CurrentPointsList]
                P_Names = [P for P in self.PointsNames if not P in CurrentPointsNames]
                if P_Names:
                    if self.MarkupVoxelMode:
                        CursorToVoxelPoint(Preffix=self.Preffix, CursorMove=True)

                    loc = context.scene.cursor.location
                    P = AddMarkupPoint(P_Names[0], self.Color, loc, self.CollName)
                    self.CurrentPointsList.append(P)

                if not P_Names:

                    Override, area3D, space3D = CtxOverride(context)
                    RefPlanes = PointsToRefPlanes(
                        Override,
                        self.TargetObject,
                        self.CurrentPointsList,
                        color=(0.0, 0.0, 0.2, 0.7),
                        CollName=self.CollName,
                    )
                    bpy.ops.object.select_all(action="DESELECT")
                    for Plane in RefPlanes:
                        Plane.select_set(True)
                    CurrentPoints = [
                        bpy.data.objects.get(PName) for PName in CurrentPointsNames
                    ]
                    for P in CurrentPoints:
                        P.select_set(True)
                    self.TargetObject.select_set(True)
                    bpy.context.view_layer.objects.active = self.TargetObject
                    bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)
                    bpy.ops.object.select_all(action="DESELECT")
                    self.DcmInfo[self.Preffix]["Frankfort"] = RefPlanes[0].name
                    self.BDENTAL_4D_Props.DcmInfo = str(self.DcmInfo)
                    ##########################################################
                    space3D.overlay.show_outline_selected = True
                    space3D.overlay.show_object_origins = True
                    space3D.overlay.show_annotation = True
                    space3D.overlay.show_text = True
                    space3D.overlay.show_extras = True
                    space3D.overlay.show_floor = True
                    space3D.overlay.show_axis_x = True
                    space3D.overlay.show_axis_y = True
                    # ###########################################################
                    bpy.ops.wm.tool_set_by_id(Override, name="builtin.select")
                    bpy.context.scene.tool_settings.use_snap = False

                    bpy.context.scene.cursor.location = (0, 0, 0)
                    # bpy.ops.screen.region_toggle(Override, region_type="UI")
                    self.BDENTAL_4D_Props.ActiveOperator = "None"

                    return {"FINISHED"}

        #########################################

        elif event.type == ("DEL") and event.value == ("PRESS"):
            if self.CurrentPointsList:
                P = self.CurrentPointsList.pop()
                bpy.data.objects.remove(P)

        elif event.type == ("ESC"):
            if self.CurrentPointsList:
                for P in self.CurrentPointsList:
                    bpy.data.objects.remove(P)

            Override, area3D, space3D = CtxOverride(context)
            ##########################################################
            space3D.overlay.show_outline_selected = True
            space3D.overlay.show_object_origins = True
            space3D.overlay.show_annotation = True
            space3D.overlay.show_text = True
            space3D.overlay.show_extras = True
            space3D.overlay.show_floor = True
            space3D.overlay.show_axis_x = True
            space3D.overlay.show_axis_y = True
            ###########################################################
            bpy.ops.wm.tool_set_by_id(Override, name="builtin.select")
            bpy.context.scene.tool_settings.use_snap = False

            bpy.context.scene.cursor.location = (0, 0, 0)
            # bpy.ops.screen.region_toggle(Override, region_type="UI")
            self.BDENTAL_4D_Props.ActiveOperator = "None"
            message = [
                " The Frankfort Plane Operation was Cancelled!",
            ]

            ShowMessageBox(message=message, icon="COLORSET_03_VEC")

            return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):

        Active_Obj = bpy.context.view_layer.objects.active

        if not Active_Obj:
            message = [" Please select Target Object ! "]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")
            return {"CANCELLED"}
        else:
            ValidTarget = Active_Obj.name.startswith("BD") and Active_Obj.name.endswith(
                ("_CTVolume", "SEGMENTATION")
            )
            if Active_Obj.select_get() == False or not ValidTarget:
                message = [
                    " Please select Target Object ! ",
                    "Target Object should be a CTVolume or a Segmentation",
                ]
                ShowMessageBox(message=message, icon="COLORSET_02_VEC")
                return {"CANCELLED"}

            else:
                if context.space_data.type == "VIEW_3D":

                    self.BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props

                    # Prepare scene  :
                    ##########################################################
                    bpy.context.space_data.overlay.show_outline_selected = False
                    bpy.context.space_data.overlay.show_object_origins = False
                    bpy.context.space_data.overlay.show_annotation = False
                    bpy.context.space_data.overlay.show_text = True
                    bpy.context.space_data.overlay.show_extras = False
                    bpy.context.space_data.overlay.show_floor = False
                    bpy.context.space_data.overlay.show_axis_x = False
                    bpy.context.space_data.overlay.show_axis_y = False
                    bpy.context.scene.tool_settings.use_snap = True
                    bpy.context.scene.tool_settings.snap_elements = {"FACE"}
                    bpy.context.scene.tool_settings.transform_pivot_point = (
                        "INDIVIDUAL_ORIGINS"
                    )
                    bpy.ops.wm.tool_set_by_id(name="builtin.cursor")

                    ###########################################################
                    self.CollName = "REFERENCE PLANES"
                    self.CurrentPointsList = []
                    self.PointsNames = ["Na", "R_Or", "R_Po", "L_Or", "L_Po"]
                    self.Color = [1, 0, 0, 1]  # Red color
                    self.TargetObject = Active_Obj
                    self.visibleObjects = bpy.context.visible_objects.copy()
                    self.MarkupVoxelMode = self.TargetObject.name.endswith("_CTVolume")
                    self.Preffix = self.TargetObject.name[:5]
                    DcmInfo = self.BDENTAL_4D_Props.DcmInfo
                    self.DcmInfo = eval(DcmInfo)
                    Override, area3D, space3D = CtxOverride(context)
                    # bpy.ops.screen.region_toggle(Override, region_type="UI")
                    bpy.ops.object.select_all(action="DESELECT")
                    # bpy.ops.object.select_all(Override, action="DESELECT")

                    context.window_manager.modal_handler_add(self)
                    self.BDENTAL_4D_Props.ActiveOperator = (
                        "bdental.add_reference_planes"
                    )
                    return {"RUNNING_MODAL"}

                else:
                    message = [
                        "Active space must be a View3d",
                    ]
                    ShowMessageBox(message=message, icon="COLORSET_02_VEC")

                    return {"CANCELLED"}


class BDENTAL_4D_OT_AddMarkupPoint(bpy.types.Operator):
    """ Add Markup point """

    bl_idname = "bdental.add_markup_point"
    bl_label = "ADD MARKUP POINT"

    MarkupName: StringProperty(
        name="Markup Name",
        default="Markup 01",
        description="Markup Name",
    )
    MarkupColor: FloatVectorProperty(
        name="Markup Color",
        description="Markup Color",
        default=[1.0, 0.0, 0.0, 1.0],
        size=4,
        subtype="COLOR",
    )

    CollName = "Markup Points"

    def execute(self, context):

        if self.MarkupVoxelMode:
            Preffix = self.TargetObject.name[:5]
            CursorToVoxelPoint(Preffix=Preffix, CursorMove=True)

        Co = context.scene.cursor.location
        P = AddMarkupPoint(
            name=self.MarkupName, color=self.MarkupColor, loc=Co, CollName=self.CollName
        )

        return {"FINISHED"}

    def invoke(self, context, event):

        self.BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props

        Active_Obj = bpy.context.view_layer.objects.active

        if not Active_Obj:
            message = [" Please select Target Object ! "]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")
            return {"CANCELLED"}

        else:
            if Active_Obj.select_get() == False:
                message = [" Please select Target Object ! "]
                ShowMessageBox(message=message, icon="COLORSET_02_VEC")
                return {"CANCELLED"}

            else:
                self.TargetObject = Active_Obj
                self.MarkupVoxelMode = self.TargetObject.name.startswith(
                    "BD"
                ) and self.TargetObject.name.endswith("_CTVolume")
                wm = context.window_manager
                return wm.invoke_props_dialog(self)


class BDENTAL_4D_OT_CtVolumeOrientation(bpy.types.Operator):
    """ CtVolume Orientation according to Frankfort Plane """

    bl_idname = "bdental.ctvolume_orientation"
    bl_label = "CTVolume Orientation"

    def execute(self, context):

        BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props
        Active_Obj = bpy.context.view_layer.objects.active

        if not Active_Obj:
            message = [" Please select CTVOLUME for segmentation ! "]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")
            return {"CANCELLED"}

        else:
            if not Active_Obj.name.startswith("BD"):
                message = [
                    "CTVOLUME Orientation : ",
                    "Please select CTVOLUME or Segmentation! ",
                ]
                ShowMessageBox(message=message, icon="COLORSET_02_VEC")
                return {"CANCELLED"}

            else:
                Preffix = Active_Obj.name[:5]
                DcmInfo = eval(BDENTAL_4D_Props.DcmInfo)
                if not "Frankfort" in DcmInfo[Preffix].keys():
                    message = [
                        "CTVOLUME Orientation : ",
                        "Please Add Reference Planes before CTVOLUME Orientation ! ",
                    ]
                    ShowMessageBox(message=message, icon="COLORSET_02_VEC")
                    return {"CANCELLED"}
                else:
                    Frankfort_Plane = bpy.data.objects.get(
                        DcmInfo[Preffix]["Frankfort"]
                    )
                    if not Frankfort_Plane:
                        message = [
                            "CTVOLUME Orientation : ",
                            "Frankfort Reference Plane has been removed",
                            "Please Add Reference Planes before CTVOLUME Orientation ! ",
                        ]
                        ShowMessageBox(message=message, icon="COLORSET_02_VEC")
                        return {"CANCELLED"}
                    else:
                        Vol = [
                            obj
                            for obj in bpy.data.objects
                            if Preffix in obj.name and "_CTVolume" in obj.name
                        ][0]
                        Vol.matrix_world = (
                            Frankfort_Plane.matrix_world.inverted() @ Vol.matrix_world
                        )
                        bpy.ops.view3d.view_center_cursor()
                        bpy.ops.view3d.view_all(center=True)
                        return {"FINISHED"}


class BDENTAL_4D_OT_ResetCtVolumePosition(bpy.types.Operator):
    """ Reset the CtVolume to its original Patient Position """

    bl_idname = "bdental.reset_ctvolume_position"
    bl_label = "RESET CTVolume POSITION"

    def execute(self, context):

        BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props
        Active_Obj = bpy.context.view_layer.objects.active

        if not Active_Obj:
            message = [" Please select CTVOLUME for segmentation ! "]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")
            return {"CANCELLED"}
        else:
            Conditions = [
                not Active_Obj.name.startswith("BD"),
                not Active_Obj.name.endswith(("_CTVolume", "SEGMENTATION")),
            ]

            if Conditions[0] or Conditions[1]:
                message = [
                    "Reset Position : ",
                    "Please select CTVOLUME or Segmentation! ",
                ]
                ShowMessageBox(message=message, icon="COLORSET_02_VEC")
                return {"CANCELLED"}

            else:
                Preffix = Active_Obj.name[:5]
                Vol = [
                    obj
                    for obj in bpy.data.objects
                    if Preffix in obj.name and "_CTVolume" in obj.name
                ][0]
                DcmInfoDict = eval(BDENTAL_4D_Props.DcmInfo)
                DcmInfo = DcmInfoDict[Preffix]
                TransformMatrix = DcmInfo["TransformMatrix"]
                Vol.matrix_world = TransformMatrix

                return {"FINISHED"}


class BDENTAL_4D_OT_AddSleeve(bpy.types.Operator):
    """ Add Sleeve """

    bl_idname = "bdental.add_sleeve"
    bl_label = "ADD SLEEVE"

    OrientationTypes = ["AXIAL", "SAGITAL/CORONAL"]
    items = []
    for i in range(len(OrientationTypes)):
        item = (str(OrientationTypes[i]), str(OrientationTypes[i]), str(""), int(i))
        items.append(item)

    Orientation: EnumProperty(items=items, description="Orientation", default="AXIAL")

    def execute(self, context):

        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=64,
            radius=self.HoleDiameter / 2 + self.HoleOffset,
            depth=30,
            align="CURSOR",
        )
        Pin = context.object
        Pin.name = "BDENTAL_4D_Pin"
        if self.Orientation == "SAGITAL/CORONAL":
            Pin.rotation_euler.rotate_axis("X", radians(-90))

        bpy.ops.mesh.primitive_cylinder_add(
            vertices=64,
            radius=self.SleeveDiametre / 2,
            depth=self.SleeveHeight,
            align="CURSOR",
        )
        Sleeve = context.object
        Sleeve.name = "BDENTAL_4D_Sleeve"
        Sleeve.matrix_world = Pin.matrix_world

        Sleeve.matrix_world.translation += Sleeve.matrix_world.to_3x3() @ Vector(
            (0, 0, self.SleeveHeight / 2)
        )

        AddMaterial(
            Obj=Pin,
            matName="BDENTAL_4D_Pin_mat",
            color=[0.0, 0.3, 0.8, 1.0],
            transparacy=None,
        )
        AddMaterial(
            Obj=Sleeve,
            matName="BDENTAL_4D_Sleeve_mat",
            color=[1.0, 0.34, 0.0, 1.0],
            transparacy=None,
        )
        Pin.select_set(True)
        context.view_layer.objects.active = Sleeve
        bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)
        Pin.select_set(False)

        return {"FINISHED"}

    def invoke(self, context, event):

        BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props
        self.SleeveDiametre = BDENTAL_4D_Props.SleeveDiameter
        self.SleeveHeight = BDENTAL_4D_Props.SleeveHeight
        self.HoleDiameter = BDENTAL_4D_Props.HoleDiameter
        self.HoleOffset = BDENTAL_4D_Props.HoleOffset
        self.cursor = context.scene.cursor

        wm = context.window_manager
        return wm.invoke_props_dialog(self)


class BDENTAL_4D_OT_AddImplant(bpy.types.Operator):
    """ Add Implant """

    bl_idname = "bdental.add_implant"
    bl_label = "ADD IMPLANT"

    OrientationTypes = ["AXIAL", "SAGITAL/CORONAL"]
    items = []
    for i in range(len(OrientationTypes)):
        item = (str(OrientationTypes[i]), str(OrientationTypes[i]), str(""), int(i))
        items.append(item)

    Orientation: EnumProperty(items=items, description="Orientation", default="AXIAL")

    D = ["3.5", "4.0", "4.5"]
    items = []
    for i in range(len(D)):
        item = (str(D[i]), str(D[i]), str(""), int(i))
        items.append(item)

    Implant_Diameter: EnumProperty(items=items, description="DIAMETER", default="4.0")

    L = ["8.5", "10", "11.5", "13"]
    items = []
    for i in range(len(L)):
        item = (str(L[i]), str(L[i]), str(""), int(i))
        items.append(item)

    Implant_Lenght: EnumProperty(items=items, description="LENGHT", default="10")

    def execute(self, context):
        cursor = context.scene.cursor
        filename = f"IMPLANT_{self.Implant_Diameter}_{self.Implant_Lenght}"
        directory = join(ImplantLibraryBlendFile, "Object")
        bpy.ops.wm.append(directory=directory, filename=filename)
        Implant = context.selected_objects[0]
        context.view_layer.objects.active = Implant
        Implant.matrix_world = cursor.matrix

        if self.Orientation == "SAGITAL/CORONAL":
            Implant.rotation_euler.rotate_axis("X", radians(-90))

        return {"FINISHED"}

    def invoke(self, context, event):

        wm = context.window_manager
        return wm.invoke_props_dialog(self)


#################################################################################################
# Registration :
#################################################################################################

classes = [
    BDENTAL_4D_OT_Template,
    BDENTAL_4D_OT_Volume_Render,
    BDENTAL_4D_OT_ResetCtVolumePosition,
    BDENTAL_4D_OT_TresholdUpdate,
    BDENTAL_4D_OT_AddSlices,
    # BDENTAL_4D_OT_TreshSegment,
    BDENTAL_4D_OT_MultiTreshSegment,
    BDENTAL_4D_OT_MultiView,
    BDENTAL_4D_OT_AddReferencePlanes,
    BDENTAL_4D_OT_CtVolumeOrientation,
    BDENTAL_4D_OT_AddMarkupPoint,
    BDENTAL_4D_OT_AddSleeve,
    BDENTAL_4D_OT_AddImplant,
]


def register():

    for cls in classes:
        bpy.utils.register_class(cls)
    post_handlers = bpy.app.handlers.depsgraph_update_post
    MyPostHandlers = [
        "BDENTAL_4D_TresholdUpdate",
        "AxialSliceUpdate",
        "CoronalSliceUpdate",
        "SagitalSliceUpdate",
    ]

    # Remove old handlers :
    handlers_To_Remove = [h for h in post_handlers if h.__name__ in MyPostHandlers]
    if handlers_To_Remove:
        for h in handlers_To_Remove:
            bpy.app.handlers.depsgraph_update_post.remove(h)

    handlers_To_Add = [
        BDENTAL_4D_TresholdUpdate,
        AxialSliceUpdate,
        CoronalSliceUpdate,
        SagitalSliceUpdate,
    ]
    for h in handlers_To_Add:
        post_handlers.append(h)
    # post_handlers.append(BDENTAL_4D_TresholdUpdate)
    # post_handlers.append(AxialSliceUpdate)
    # post_handlers.append(CoronalSliceUpdate)
    # post_handlers.append(SagitalSliceUpdate)


def unregister():

    post_handlers = bpy.app.handlers.depsgraph_update_post
    MyPostHandlers = [
        "BDENTAL_4D_TresholdUpdate",
        "AxialSliceUpdate",
        "CoronalSliceUpdate",
        "SagitalSliceUpdate",
    ]
    handlers_To_Remove = [h for h in post_handlers if h.__name__ in MyPostHandlers]

    if handlers_To_Remove:
        for h in handlers_To_Remove:
            bpy.app.handlers.depsgraph_update_post.remove(h)

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
