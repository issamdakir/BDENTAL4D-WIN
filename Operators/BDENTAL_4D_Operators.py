import os, stat, sys, shutil, math, threading,pickle,glob
from math import degrees, radians, pi, ceil, floor, sqrt
import numpy as np
from time import sleep, perf_counter as Tcounter
from queue import Queue
from os.path import join, dirname, abspath, exists, split, basename
from importlib import reload

import gpu
from gpu_extras.batch import batch_for_shader
import bgl
import blf

# Blender Imports :
import bpy, bpy_extras
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

from vtk.util import numpy_support
from vtk import vtkCommand

# Global Variables :

# from . import BDENTAL_4D_Utils
from .BDENTAL_4D_Utils import *

Addon_Enable(AddonName="mesh_looptools", Enable=True)

addon_dir = dirname(dirname(abspath(__file__)))
DataBlendFile = join(addon_dir, "Resources", "BlendData", "BDENTAL4D_BlendData.blend")

GpShader = "VGS_Marcos_modified_MinMax"#"VGS_Marcos_modified"  # "VGS_Marcos_01" "VGS_Dakir_01"
Wmin = -400
Wmax = 3000
ProgEvent = vtkCommand.ProgressEvent
#######################################################################################
########################### CT Scan Load : Operators ##############################
#######################################################################################
class BDENTAL_4D_OT_OpenManual(bpy.types.Operator):
    """ Open BDENTAL_4D Manual """

    bl_idname = "bdental4d.open_manual"
    bl_label = "User Manual"

    def execute(self, context):

        Manual_Path = join(addon_dir, "Resources", "BDENTAL4D User Manual.pdf")
        os.startfile(Manual_Path)
        return {"FINISHED"}


class BDENTAL_4D_OT_Template(bpy.types.Operator):
    """ Open BDENTAL_4D workspace template """

    bl_idname = "bdental4d.template"
    bl_label = "OPEN BDENTAL_4D WORKSPACE"

    SaveMainFile: BoolProperty(description="Save Main File", default=False)
    UserProjectDir: StringProperty(
        name="INFO : ",
        default=" Please Restart Blender after",
        description="",
    )
    Themes = [ basename(f).split('.')[0] for f in glob.glob(join(addon_dir,'Resources', '*')) if f.endswith('xml')]
    items = []
    for i in range(len(Themes)):
        item = (str(Themes[i]), str(Themes[i]), str(""), int(i))
        items.append(item)

    ThemesProp : EnumProperty(items=items, description="Voxel Mode", default=Themes[0])

    def execute(self, context):

        CurrentBlendFile = bpy.path.abspath(bpy.data.filepath)
        BlendStartFile = join(
            addon_dir, "Resources", "BlendData", "BlendStartFile.blend"
        )

        # Install or load BDENTAL_4D theme :
        ScriptsPath = dirname(dirname(addon_dir))
        BDENTAL_4D_Theme_installed = join(
            ScriptsPath, "presets", "interface_theme", self.ThemesProp+".xml"
        )
        if not exists(BDENTAL_4D_Theme_installed):
            BDENTAL_4D_Theme = join(addon_dir, "Resources", self.ThemesProp+".xml")
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


class BDENTAL_4D_OT_Organize(bpy.types.Operator):
    """ DICOM Organize """

    bl_idname = "bdental4d.organize"
    bl_label = "ORGANIZE DICOM"

    def execute(self, context):
        BDENTAL_4D_Props = context.scene.BDENTAL_4D_Props
        UserProjectDir = AbsPath(BDENTAL_4D_Props.UserProjectDir)
        UserDcmDir = AbsPath(BDENTAL_4D_Props.UserDcmDir)

        DcmOrganizeDict = eval(BDENTAL_4D_Props.DcmOrganize)

        if UserDcmDir in DcmOrganizeDict.keys() :
            OrganizeReport = DcmOrganizeDict[UserDcmDir]

        else :
            if not exists(UserProjectDir):
                message = [" The Selected Project Directory is not valid ! "]
                ShowMessageBox(message=message, icon="COLORSET_02_VEC")
                return {"CANCELLED"}

            if not exists(UserDcmDir):
                message = [" The Selected Dicom Directory is not valid ! "]
                ShowMessageBox(message=message, icon="COLORSET_02_VEC")
                return {"CANCELLED"}

            if not os.listdir(UserDcmDir):
                message = ["No DICOM files found in DICOM Folder ! "]
                ShowMessageBox(message=message, icon="COLORSET_02_VEC")
                return {"CANCELLED"}

            Series_reader = sitk.ImageSeriesReader()
            Series_IDs = Series_reader.GetGDCMSeriesIDs(UserDcmDir)
            
            
            if not Series_IDs:
                message = ["No valid DICOM Serie found in DICOM Folder ! "]
                print(message)
                ShowMessageBox(message=message, icon="COLORSET_01_VEC")
                return {"CANCELLED"}

            series_IDs_Files = [[S_ID,Series_reader.GetGDCMSeriesFileNames(UserDcmDir, S_ID)] for S_ID in Series_IDs]

            tags = dict(
                    {
                        "Patient Name": "0010|0010",
                        "Series Date": "0008|0021",
                        "Series Description": "0008|103E",
                        
            
                    }
                )
            DcmOrganizeDict[UserDcmDir]= {}
            for [S_ID,FilesList] in series_IDs_Files:
                count = len(FilesList)
                DcmOrganizeDict[UserDcmDir][S_ID]={'Count':count}#,'Files':FilesList}

                file0 = FilesList[0]
                reader = sitk.ImageFileReader()
                reader.SetFileName(file0)
                reader.LoadPrivateTagsOn()
                reader.ReadImageInformation()

                Image = reader.Execute()

                for attribute, tag in tags.items():

                    if tag in Image.GetMetaDataKeys():
                        v = Image.GetMetaData(tag)

                    else:
                        v = "-------"

                    DcmOrganizeDict[UserDcmDir][S_ID][attribute] = v

                DcmOrganizeDict[UserDcmDir][S_ID]['Files'] = FilesList


            SortedList = sorted(DcmOrganizeDict[UserDcmDir].items(), key=lambda x: x[1]['Count'], reverse=True)
            # Sorted = list(sorted(DcmOrganizeDict[UserDcmDir], reverse=True))

            SortedOrganizeDict = {}
            for i,(k,v) in enumerate(SortedList) :
                SortedOrganizeDict[f"Series{i} ({v['Count']})"] = DcmOrganizeDict[UserDcmDir][k]
            # for k,v in SortedOrganizeDict.items():
            #     print(k,' : ',v['Count'])
            DcmOrganizeDict[UserDcmDir] = SortedOrganizeDict
            BDENTAL_4D_Props.DcmOrganize = str(DcmOrganizeDict)
            OrganizeReport = SortedOrganizeDict

        Message = {}
        for serie, info in OrganizeReport.items() :
            Count, Name, Date, Descript = info['Count'], info['Patient Name'], info['Series Date'], info['Series Description']
            Message[serie]={'Count': Count,
                            'Patient Name': Name,
                            'Series Date': Date,
                            'Series Description': Descript}
        
        BDENTAL_4D_Props.OrganizeInfoProp = str(Message)

        ProjectName = BDENTAL_4D_Props.ProjectNameProp

        # Save Blend File :
        BlendFile = f"{ProjectName}.blend"
        Blendpath = join(UserProjectDir , BlendFile)
        bpy.ops.wm.save_as_mainfile(filepath=Blendpath)
        BDENTAL_4D_Props.UserProjectDir = RelPath(UserProjectDir)
        bpy.ops.wm.save_mainfile()

        # Split = split(UserProjectDir )
        # ProjectName = Split[-1] or Split[-2]
        # BlendFile = f"{ProjectName}_CT-SCAN.blend"
        # Blendpath = join(UserProjectDir , BlendFile)

        # bpy.ops.wm.save_as_mainfile(filepath=Blendpath)
        # BDENTAL_4D_Props.UserProjectDir = RelPath(UserProjectDir)
        # bpy.ops.wm.save_mainfile()

        # file1 = join(UserDcmDir, os.listdir(UserDcmDir)[1])
        # reader = sitk.ImageFileReader()
        # reader.SetFileName(file1)
        # reader.LoadPrivateTagsOn()
        # reader.ReadImageInformation()
        # Image = reader.Execute()

        # ConvKernel = None
        # if "0018|1210" in Image.GetMetaDataKeys():
        #     ConvKernel = Image.GetMetaData("0018|1210")
        #     if not ConvKernel:
        #         ConvKernel = None
        # Manufacturer = None
        # if "0008|0070" in Image.GetMetaDataKeys():
        #     Manufacturer = Image.GetMetaData("0008|0070")
        #     if not Manufacturer:
        #         Manufacturer = None
        
        # Soft,Bone,Teeth = GetAutoReconstructParameters(Manufacturer, ConvKernel)

        # print(Soft,Bone,Teeth)
        
        return {'FINISHED'}

def Load_Dicom_funtion(context, q):

    message = []

    ################################################################################################
    start = Tcounter()
    ################################################################################################
    BDENTAL_4D_Props = context.scene.BDENTAL_4D_Props
    UserProjectDir = AbsPath(BDENTAL_4D_Props.UserProjectDir)

    Serie = BDENTAL_4D_Props.Dicom_Series
    # Start Reading Dicom data :
    ######################################################################################
    UserDcmDir = AbsPath(BDENTAL_4D_Props.UserDcmDir)
    DcmOrganizeDict = eval(BDENTAL_4D_Props.DcmOrganize)
    DcmSerie = DcmOrganizeDict[UserDcmDir][Serie]['Files']
    Image3D = sitk.ReadImage(DcmSerie)  
        
    # Get Preffix and save file :
    DcmInfoDict = eval(BDENTAL_4D_Props.DcmInfo)
    Preffixs = list(DcmInfoDict.keys())

    for i in range(1, 100):
        Preffix = f"BD4D_{i:03}"
        if not Preffix in Preffixs:
            break

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

    DcmInfo = dict(
        {
            "SlicesDir" : "",
            "Nrrd255Path":"",
            "Preffix": Preffix,
            "RenderSz": Sz,
            "RenderSp": Sp,
            "PixelType": Image3D.GetPixelIDTypeAsString(),
            "Wmin": Wmin,
            "Wmax": Wmax,
            "Dims": Dims,
            "Size": Sz,
            "Spacing": Sp,
            "Origin": Origin,
            "Direction": Direction,
            "TransformMatrix": TransformMatrix,
            "DirectionMatrix_4x4": DirectionMatrix_4x4,
            "TransMatrix_4x4": TransMatrix_4x4,
            "VtkTransform_4x4": VtkTransform_4x4,
            "VolumeCenter": VCenter,
            'AutoReconParameters' : '[None,None,None]'
        }
    )
    # Get Automatic Reconstruction Parameters :
    reader = sitk.ImageFileReader()
    reader.SetFileName(DcmSerie[0])
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    ImageReader = reader.Execute()

    ConvKernel = None
    if "0018|1210" in ImageReader.GetMetaDataKeys():
        ConvKernel = ImageReader.GetMetaData("0018|1210")
        if not ConvKernel:
            ConvKernel = None
    Manufacturer = None
    if "0008|0070" in ImageReader.GetMetaDataKeys():
        Manufacturer = ImageReader.GetMetaData("0008|0070")
        if not Manufacturer:
            Manufacturer = None
    
    Soft,Bone,Teeth = GetAutoReconstructParameters(Manufacturer, ConvKernel)
    DcmInfo['AutoReconParameters']=str([Soft,Bone,Teeth])
    print(DcmInfo['AutoReconParameters'])
    #######################################################################################
    # Add directories :
    PngDir = join(UserProjectDir ,"PNG")
    AxialPngDir = join(PngDir, "Axial")
    CoronalPngDir = join(PngDir, "Coronal")
    SagitalPngDir = join(PngDir, "Sagital")

    os.makedirs(AxialPngDir)
    os.makedirs(CoronalPngDir)
    os.makedirs(SagitalPngDir)



    Nrrd255Path = join(UserProjectDir , f"{Preffix}_Image3D255.nrrd")
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

    

    

    #############################################################################################
    # MultiThreading PNG Writer:
    #########################################################################################
    def Image3DToAxialPNG(i, slices, AxialPngDir, Preffix):
        img_Slice = slices[i]
        img_Name = f"{Preffix}_Axial_img{i:04}.png"
        image_path = join(AxialPngDir, img_Name)
        cv2.imwrite(image_path, img_Slice)
        image = bpy.data.images.load(image_path)
        image.pack()
        # print(f"{img_Name} was processed...")
    def Image3DToCoronalPNG(i, slices, CoronalPngDir, Preffix):
        img_Slice = slices[i]
        img_Name = f"{Preffix}_Coronal_img{i:04}.png"
        image_path = join(CoronalPngDir, img_Name)
        cv2.imwrite(image_path, img_Slice)
        image = bpy.data.images.load(image_path)
        image.pack()
        # print(f"{img_Name} was processed...")
    def Image3DToSagitalPNG(i, slices, SagitalPngDir, Preffix):
        img_Slice = slices[i]
        img_Name = f"{Preffix}_Sagital_img{i:04}.png"
        image_path = join(SagitalPngDir, img_Name)
        cv2.imwrite(image_path, img_Slice)
        image = bpy.data.images.load(image_path)
        image.pack()
        # print(f"{img_Name} was processed...")

    #########################################################################################
    # Get slices list :
    
    MaxSp = max(Vector(Sp))
    if MaxSp < 0.25:
        SampleRatio = round(MaxSp / 0.25, 2)
        Image3D_255, new_size, new_spacing = ResizeImage(sitkImage=Image3D_255, Ratio=SampleRatio)
        DcmInfo["RenderSz"] = new_size
        DcmInfo["RenderSp"] = new_spacing
        print('image resized for speed render')

    # Convert Dicom to nrrd file :
    sitk.WriteImage(Image3D_255, Nrrd255Path)

    # make axial slices : 
    Array = sitk.GetArrayFromImage(Image3D_255)
    AxialSlices = [np.flipud(Array[i, :, :]) for i in range(Array.shape[0])]
    CoronalSlices = [np.flipud(Array[:, i, :]) for i in range(Array.shape[1])]
    SagitalSlices = [np.flipud(Array[:, :, i]) for i in range(Array.shape[2])]
    # slices = [Image3D_255[:, :, i] for i in range(Image3D_255.GetDepth())]

    Axialthreads = [
        threading.Thread(
            target=Image3DToAxialPNG,
            args=[i, AxialSlices, AxialPngDir, Preffix],
            daemon=True,
        )
        for i in range(len(AxialSlices))
    ]
    Coronalthreads = [
        threading.Thread(
            target=Image3DToCoronalPNG,
            args=[i, CoronalSlices, CoronalPngDir, Preffix],
            daemon=True,
        )
        for i in range(len(CoronalSlices))
    ]
    Sagitalthreads = [
        threading.Thread(
            target=Image3DToSagitalPNG,
            args=[i, SagitalSlices, SagitalPngDir, Preffix],
            daemon=True,
        )
        for i in range(len(SagitalSlices))
    ]
    threads = Axialthreads+Coronalthreads+Sagitalthreads
    for t in threads:
        t.start()

    for t in threads:
        t.join()
    
    shutil.rmtree(PngDir)
    
    DcmInfo["CT_Loaded"] = True
    # Set DcmInfo property :
    DcmInfoDict = eval(BDENTAL_4D_Props.DcmInfo)
    DcmInfoDict[Preffix] = DcmInfo
    
    BDENTAL_4D_Props.DcmInfo = str(DcmInfoDict)
    
    return DcmInfo, message
####### End Load_Dicom_fuction ##############


#######################################################################################
# BDENTAL_4D CT Scan 3DImage File Load :


def Load_3DImage_function(context, q):
    DcmInfo, message = [],[]
    BDENTAL_4D_Props = context.scene.BDENTAL_4D_Props
    UserProjectDir = AbsPath(BDENTAL_4D_Props.UserProjectDir)
    UserImageFile = AbsPath(BDENTAL_4D_Props.UserImageFile)
    ProjectName = BDENTAL_4D_Props.ProjectNameProp

    # Save Blend File :
    BlendFile = f"{ProjectName}.blend"
    Blendpath = join(UserProjectDir , BlendFile)
    
    bpy.ops.wm.save_as_mainfile(filepath=Blendpath)

    BDENTAL_4D_Props.UserProjectDir = RelPath(UserProjectDir)
    bpy.ops.wm.save_mainfile()

    reader = sitk.ImageFileReader()
    IO = reader.GetImageIOFromFileName(UserImageFile)
    FileExt = os.path.splitext(UserImageFile)[1]

    if not IO:
        message = [
            f"{FileExt} files are not Supported! for more info about supported files please refer to Addon wiki "
        ]
        return DcmInfo, message
        

    Image3D = sitk.ReadImage(UserImageFile)
    Depth = Image3D.GetDepth()
    print(f'Depth : {Depth}')

    if Depth <= 1:
        message = [
            "Can't Build 3D Volume from 2D Image !",
            "for more info about supported files,",
            "please refer to Addon wiki",
        ]
        return DcmInfo, message
        

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
        return DcmInfo, message
    ###########################################################################################################

    else:

        start = Tcounter()
        ####################################
        # Get Preffix and save file :
        DcmInfoDict = eval(BDENTAL_4D_Props.DcmInfo)
        Preffixs = list(DcmInfoDict.keys())

        for i in range(1, 100):
            Preffix = f"BD4D_{i:03}"
            if not Preffix in Preffixs:
                break

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

        DcmInfo = dict(
            {
                "SlicesDir" : "",
                "Nrrd255Path":"",
                "Preffix": Preffix,
                "RenderSz": Sz,
                "RenderSp": Sp,
                "PixelType": Image3D.GetPixelIDTypeAsString(),
                "Wmin": Wmin,
                "Wmax": Wmax,
                "Dims": Dims,
                "Size": Sz,
                "BackupArray": "",
                "Spacing": Sp,
                "Origin": Origin,
                "Direction": Direction,
                "TransformMatrix": TransformMatrix,
                "DirectionMatrix_4x4": DirectionMatrix_4x4,
                "TransMatrix_4x4": TransMatrix_4x4,
                "VtkTransform_4x4": VtkTransform_4x4,
                "VolumeCenter": VCenter,
                'AutoReconParameters' : '[None,None,None]'

            }
        )

        

        #######################################################################################
        # Add directories :
        PngDir = join(UserProjectDir ,"PNG")
        AxialPngDir = join(PngDir, "Axial")
        CoronalPngDir = join(PngDir, "Coronal")
        SagitalPngDir = join(PngDir, "Sagital")

        os.makedirs(AxialPngDir)
        os.makedirs(CoronalPngDir)
        os.makedirs(SagitalPngDir)



        Nrrd255Path = join(AbsPath(UserProjectDir) , f"{Preffix}_Image3D255.nrrd")
        DcmInfo["Nrrd255Path"] = RelPath(Nrrd255Path)

        #######################################################################################
        if BDENTAL_4D_nrrd :
            Image3D_255 = Image3D
        else : 
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
        sitk.WriteImage(Image3D_255, Nrrd255Path)
        # BackupArray = sitk.GetArrayFromImage(Image3D_255)
        # DcmInfo["BackupArray"] = str(BackupArray.tolist())


        #############################################################################################
        # MultiThreading PNG Writer:
        #########################################################################################
        def Image3DToAxialPNG(i, slices, AxialPngDir, Preffix):
            img_Slice = slices[i]
            img_Name = f"{Preffix}_Axial_img{i:04}.png"
            image_path = join(AxialPngDir, img_Name)
            cv2.imwrite(image_path, img_Slice)
            image = bpy.data.images.load(image_path)
            image.pack()
            # print(f"{img_Name} was processed...")
        def Image3DToCoronalPNG(i, slices, CoronalPngDir, Preffix):
            img_Slice = slices[i]
            img_Name = f"{Preffix}_Coronal_img{i:04}.png"
            image_path = join(CoronalPngDir, img_Name)
            cv2.imwrite(image_path, img_Slice)
            image = bpy.data.images.load(image_path)
            image.pack()
            # print(f"{img_Name} was processed...")
        def Image3DToSagitalPNG(i, slices, SagitalPngDir, Preffix):
            img_Slice = slices[i]
            img_Name = f"{Preffix}_Sagital_img{i:04}.png"
            image_path = join(SagitalPngDir, img_Name)
            cv2.imwrite(image_path, img_Slice)
            image = bpy.data.images.load(image_path)
            image.pack()
            # print(f"{img_Name} was processed...")

        #########################################################################################
        # Get slices list :
        MaxSp = max(Vector(Sp))
        if MaxSp < 0.25:
            SampleRatio = round(MaxSp / 0.25, 2)
            Image3D_255, new_size, new_spacing = ResizeImage(sitkImage=Image3D_255, Ratio=SampleRatio)
            DcmInfo["RenderSz"] = new_size
            DcmInfo["RenderSp"] = new_spacing
            print('image resized for speed render')

        # make axial slices : 
        Array = sitk.GetArrayFromImage(Image3D_255)
        AxialSlices = [np.flipud(Array[i, :, :]) for i in range(Array.shape[0])]
        CoronalSlices = [np.flipud(Array[:, i, :]) for i in range(Array.shape[1])]
        SagitalSlices = [np.flipud(Array[:, :, i]) for i in range(Array.shape[2])]
        # slices = [Image3D_255[:, :, i] for i in range(Image3D_255.GetDepth())]

        Axialthreads = [
            threading.Thread(
                target=Image3DToAxialPNG,
                args=[i, AxialSlices, AxialPngDir, Preffix],
                daemon=True,
            )
            for i in range(len(AxialSlices))
        ]
        Coronalthreads = [
            threading.Thread(
                target=Image3DToCoronalPNG,
                args=[i, CoronalSlices, CoronalPngDir, Preffix],
                daemon=True,
            )
            for i in range(len(CoronalSlices))
        ]
        Sagitalthreads = [
            threading.Thread(
                target=Image3DToSagitalPNG,
                args=[i, SagitalSlices, SagitalPngDir, Preffix],
                daemon=True,
            )
            for i in range(len(SagitalSlices))
        ]
        threads = Axialthreads+Coronalthreads+Sagitalthreads
        for t in threads:
            t.start()

        for t in threads:
            t.join()

        
        shutil.rmtree(AxialPngDir)
        shutil.rmtree(CoronalPngDir)
        shutil.rmtree(SagitalPngDir)
        DcmInfo["CT_Loaded"] = True
        # Set DcmInfo property :
        DcmInfoDict = eval(BDENTAL_4D_Props.DcmInfo)
        DcmInfoDict[Preffix] = DcmInfo
        BDENTAL_4D_Props.DcmInfo = str(DcmInfoDict)

        return DcmInfo, message


##########################################################################################
######################### BDENTAL_4D Volume Render : ########################################
##########################################################################################
class BDENTAL_4D_OT_Volume_Render(bpy.types.Operator):
    """ Volume Render """

    bl_idname = "bdental4d.volume_render"
    bl_label = "VOXEL 3D"

    q = Queue()
    Voxel_Modes = ["FAST", "OPTIMAL", "FULL"]
    items = []
    for i in range(len(Voxel_Modes)):
        item = (str(Voxel_Modes[i]), str(Voxel_Modes[i]), str(""), int(i))
        items.append(item)

    VoxelMode : EnumProperty(items=items, description="Voxel Mode", default="FAST")


    def execute(self, context):

        Start = Tcounter()
        print("Data Loading START...")

        global DataBlendFile
        global GpShader

        BDENTAL_4D_Props = context.scene.BDENTAL_4D_Props

        DataType = BDENTAL_4D_Props.DataType

        if DataType == "DICOM Series":
            Serie = BDENTAL_4D_Props.Dicom_Series
            if not 'Series' in Serie:

                message = [" Please Organize DICOM data and retry ! "]
                ShowMessageBox(message=message, icon="COLORSET_01_VEC")
                return {"CANCELLED"}

            else:
                # Start Reading Dicom data :
                ######################################################################################
                UserDcmDir = AbsPath(BDENTAL_4D_Props.UserDcmDir)
                DcmOrganizeDict = eval(BDENTAL_4D_Props.DcmOrganize)
                DcmSerie = DcmOrganizeDict[UserDcmDir][Serie]['Files']
                DcmInfo, message = Load_Dicom_funtion(context, self.q)
                
                
                
        if DataType == "3D Image File":
            UserImageFile = AbsPath(BDENTAL_4D_Props.UserImageFile)
            if not exists(UserImageFile):
                message = [" The Selected Image File Path is not valid ! "]
                ShowMessageBox(message=message, icon="COLORSET_02_VEC")
                return {"CANCELLED"}
            
            DcmInfo, message = Load_3DImage_function(context, self.q)

        if message :
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")
            return {"CANCELLED"}
        else :    

            Preffix = DcmInfo["Preffix"]
            Wmin = DcmInfo["Wmin"]
            Wmax = DcmInfo["Wmax"]
            # PngDir = AbsPath(BDENTAL_4D_Props.PngDir)
            print("\n##########################\n")
            VolumeRender(DcmInfo, GpShader, DataBlendFile,self.VoxelMode)
            print("setting volumes...")
            scn = bpy.context.scene
            scn.render.engine = "BLENDER_EEVEE"
            BDENTAL_4D_Props.GroupNodeName = GpShader

            GpNode = bpy.data.node_groups.get(f"{Preffix}_{GpShader}")
            Low_Treshold = GpNode.nodes["Low_Treshold"].outputs[0]
            Low_Treshold.default_value = 600
            WminNode = GpNode.nodes["WminNode"].outputs[0]
            WminNode.default_value = Wmin
            WmaxNode = GpNode.nodes["WmaxNode"].outputs[0]
            WmaxNode.default_value = Wmax

            
            Soft,Bone,Teeth = eval(DcmInfo['AutoReconParameters'])
            if Soft and Bone and Teeth :
                BDENTAL_4D_Props.SoftTreshold = Soft
                BDENTAL_4D_Props.BoneTreshold = Bone
                BDENTAL_4D_Props.TeethTreshold = Teeth
                BDENTAL_4D_Props.SoftBool = True
                BDENTAL_4D_Props.BoneBool = True
                BDENTAL_4D_Props.TeethBool = True

            BDENTAL_4D_Props.CT_Rendered = True
            # bpy.ops.view3d.view_selected(use_all_regions=False)
            bpy.ops.wm.save_mainfile()
            print('Blend file path 2nd: ', bpy.data.filepath)       


            message = ["DICOM loaded successfully. "]
            print(message)
            ShowMessageBox(message=message, icon="COLORSET_03_VEC")

            
            Finish = Tcounter()
            print(f"Finished (Time : {Finish-Start}")

            return {"FINISHED"}

    def invoke(self, context, event):

        wm = context.window_manager
        return wm.invoke_props_dialog(self)
        

    

##########################################################################################
######################### BDENTAL_4D Add Slices : ########################################
##########################################################################################


class BDENTAL_4D_OT_AddSlices(bpy.types.Operator):
    """ Add Volume Slices """

    bl_idname = "bdental4d.addslices"
    bl_label = "SLICE VOLUME"

    ShowMPR : BoolProperty(description=" MPR ", default=False)

    def execute(self, context):
        BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props

        Active_Obj = context.view_layer.objects.active

        if not Active_Obj:
            message = [" Please select CTVOLUME ! "]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")
            return {"CANCELLED"}
        else:
            N = Active_Obj.name
            Condition = "BD4D" in N and "_CTVolume" in N 

            if not Condition:
                message = [" Please select CTVOLUME ! "]
                ShowMessageBox(message=message, icon="COLORSET_02_VEC")
                return {"CANCELLED"}
            else:
                Vol = Active_Obj
                Preffix = Vol.name[:8]
                DcmInfoDict = eval(BDENTAL_4D_Props.DcmInfo)
                DcmInfo = DcmInfoDict[Preffix]

                SlicesDir  = AbsPath(BDENTAL_4D_Props.SlicesDir)
                if not exists(SlicesDir) :
                    SlicesDir = tempfile.mkdtemp()
                    BDENTAL_4D_Props.SlicesDir = SlicesDir
                

                Nrrd255Path = AbsPath(DcmInfo["Nrrd255Path"])
                if not exists(Nrrd255Path) :
                    message = [" Can't find dicom data!", " Check for nrrd file in the Project Directory ! "]
                    ShowMessageBox(message=message, icon="COLORSET_02_VEC")
                    return {"CANCELLED"}
                
                SLICES_Coll = bpy.data.collections.get('SLICES')
                if SLICES_Coll :
                    SLICES_Coll.hide_viewport = False

                AxialPlane = AddAxialSlice(Preffix, DcmInfo, SlicesDir)
                MoveToCollection(obj=AxialPlane, CollName="SLICES")

                CoronalPlane = AddCoronalSlice(Preffix, DcmInfo, SlicesDir)
                MoveToCollection(obj=CoronalPlane, CollName="SLICES")

                SagitalPlane = AddSagitalSlice(Preffix, DcmInfo, SlicesDir)
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
                rotation_euler = Euler((0.0, 0.0, pi / 2), "XYZ")
                RotMtx = rotation_euler.to_matrix().to_4x4()
                SagitalCam.matrix_world = SagitalCam.matrix_world @ RotMtx

                SLICES_CAMERAS_Coll = bpy.data.collections.get("SLICES-CAMERAS")
                SLICES_CAMERAS_Coll.hide_viewport = False

                for obj in bpy.data.objects:
                    if obj.name == f"{Preffix}_SLICES_POINTER":
                        bpy.data.objects.remove(obj)

                bpy.ops.object.empty_add(
                    type="PLAIN_AXES",
                    scale=(1, 1, 1),
                )

                SLICES_POINTER = bpy.context.object
                SLICES_POINTER.matrix_world = AxialPlane.matrix_world

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

                if self.ShowMPR :
                    bpy.ops.bdental4d.mpr()
                return {"FINISHED"}

    def invoke(self, context, event):

        wm = context.window_manager
        return wm.invoke_props_dialog(self)

class BDENTAL_4D_OT_MPR(bpy.types.Operator):
    """ MultiView Toggle """

    bl_idname = "bdental4d.mpr"
    bl_label = "MPR"

    def execute(self, context):

        Active_Obj = bpy.context.view_layer.objects.active

        if not Active_Obj:
            message = [" Please select SLICES_POINTER "]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")
            return {"CANCELLED"}
        else:
            N = Active_Obj.name
            Condition = (
                N.startswith("BD4D")
                and "_SLICES_POINTER" in N
                and Active_Obj.select_get() == True
            )

            if not Condition:
                List = [o for o in context.scene.objects if '_SLICES_POINTER' in o.name]
                if List :
                    message = [
                        " Please select SLICES_POINTER ! "
                    ]
                else :
                    message = [
                        "To Add Multi-View Window :",
                        "1 - Please select CTVOLUME",
                        "2 - Click on < SLICE VOLUME > button",
                        "AXIAL, CORONAL and SAGITAL slices will be added",
                        "3 - Ensure SLICES_POINTER is Selected",
                        "3 - Click <MPR> button",
                    ]
                ShowMessageBox(message=message, icon="COLORSET_02_VEC")
                return {"CANCELLED"}
            else:
                Preffix = Active_Obj.name[:8]
                AxialPlane = bpy.data.objects.get(f"1_{Preffix}_AXIAL_SLICE")
                CoronalPlane = bpy.data.objects.get(f"2_{Preffix}_CORONAL_SLICE")
                SagitalPlane = bpy.data.objects.get(f"3_{Preffix}_SAGITAL_SLICE")
                SLICES_POINTER = Active_Obj

                if not AxialPlane or not CoronalPlane or not SagitalPlane:
                    message = [
                        "To Add Multi-View Window :",
                        "1 - Please select CTVOLUME",
                        "2 - Click on < SLICE VOLUME > button",
                        "AXIAL, CORONAL and SAGITAL slices will be added",
                        "3 - Ensure SLICES_POINTER is Selected",
                        "3 - Click <MPR> button",
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

###############################################################################
####################### BDENTAL_4D_FULL VOLUME to Mesh : ################################
##############################################################################
class BDENTAL_4D_OT_MultiTreshSegment(bpy.types.Operator):
    """ Add a mesh Segmentation using Treshold """

    bl_idname = "bdental4d.multitresh_segment"
    bl_label = "SEGMENTATION"

    TimingDict = {}

    def ImportMeshStl(self, Segment, SegmentStlPath, SegmentColor):

        # import stl to blender scene :
        bpy.ops.import_mesh.stl(filepath=SegmentStlPath)
        obj = bpy.context.object
        obj.name = f"{self.Preffix}_{Segment}_SEGMENTATION"
        obj.data.name = f"{self.Preffix}_{Segment}_mesh"

        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")

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

        print(f"{Segment} Mesh Import Finished")

        return obj

        # self.q.put(["End"])

    def DicomToStl(self, Segment, Image3D):
        print(f"{Segment} processing ...")
        # Load Infos :
        #########################################################################
        BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props
        UserProjectDir  = AbsPath(BDENTAL_4D_Props.UserProjectDir )
        DcmInfo = self.DcmInfo
        Origin = DcmInfo["Origin"]
        VtkTransform_4x4 = DcmInfo["VtkTransform_4x4"]
        TransformMatrix = DcmInfo["TransformMatrix"]
        VtkMatrix_4x4 = (
            self.Vol.matrix_world @ TransformMatrix.inverted() @ VtkTransform_4x4
        )

        VtkMatrix = list(np.array(VtkMatrix_4x4).ravel())

        
        Thikness = 1

        SegmentTreshold = self.SegmentsDict[Segment]["Treshold"]
        SegmentColor = self.SegmentsDict[Segment]["Color"]
        SegmentStlPath = join(UserProjectDir , f"{Segment}_SEGMENTATION.stl")

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

        

        self.step2 = Tcounter()
        self.TimingDict["Mesh Extraction Time"] = self.step2 - self.step1
        print(f"{Segment} Mesh Extraction Finished")

        # ############### step 3 : mesh Smoothing 1... #########################
        # SmthIter = 3
        # SmoothedMesh1 = vtkSmoothMesh(
        #     q=self.q,
        #     mesh=Mesh,
        #     Iterations=SmthIter,
        #     step="Mesh Smoothing 2",
        #     start=0.79,
        #     finish=0.82,
        # )
        # Mesh = SmoothedMesh1


        self.step3 = Tcounter()
        self.TimingDict["Mesh Smoothing 1 Time"] = self.step3 - self.step1
        print(f"{Segment} Mesh Smoothing 1 Finished")

        # ############### step 4 : mesh Smoothing... #########################

        SmthIter = 20
        SmoothedMesh2 = vtkWindowedSincPolyDataFilter(
            q=self.q,
            mesh=Mesh,
            Iterations=SmthIter,
            step="Mesh Smoothing 1",
            start=0.76,
            finish=0.78,
        )

        self.step4 = Tcounter()
        self.TimingDict["Mesh Smoothing 2 Time"] = self.step4 - self.step3
        print(f"{Segment} Mesh Smoothing 2 Finished")
        Mesh = SmoothedMesh2
        
        ############### step 5 : mesh Reduction... #########################
        polysCount = Mesh.GetNumberOfPolys()
        polysLimit = 300000
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

        self.step5 = Tcounter()
        self.TimingDict["Mesh Reduction Time"] = self.step5 - self.step4
        print(f"{Segment} Mesh Reduction Finished")
       
        
        ############### step 6 : Set mesh orientation... #########################
        TransformedMesh = vtkTransformMesh(
            mesh=Mesh,
            Matrix=VtkMatrix,
        )
        self.step6 = Tcounter()
        self.TimingDict["Mesh Orientation"] = self.step6 - self.step5
        print(f"{Segment} Mesh Orientation Finished")
        Mesh = TransformedMesh

        ############### step 7 : exporting mesh stl... #########################
        writer = vtk.vtkSTLWriter()
        writer.SetInputData(Mesh)
        writer.SetFileTypeToBinary()
        writer.SetFileName(SegmentStlPath)
        writer.Write()

        self.step7 = Tcounter()
        self.TimingDict["Mesh Export"] = self.step7 - self.step6
        print(f"{Segment} Mesh Export Finished")
        self.Exported.put([Segment, SegmentStlPath, SegmentColor])

    def execute(self, context):

        self.counter_start = Tcounter()

        BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props
        Active_Obj = bpy.context.view_layer.objects.active

        if not bpy.context.view_layer.objects.active :
            message = [" Please select CTVOLUME for segmentation ! "]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")
            return {"CANCELLED"}
        else:
            N = Active_Obj.name
            

            if not "BD4D" in N and not "_CTVolume" in N:
                message = [" Please select CTVOLUME for segmentation ! "]
                ShowMessageBox(message=message, icon="COLORSET_02_VEC")
                return {"CANCELLED"}

            else:
                print('Active object name :',N)

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
                    self.Preffix = self.Vol.name[:8]
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
                            ResizedImage,_,_ = ResizeImage(
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
                                os.remove(SegmentStlPath)
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
                        bpy.ops.object.shade_flat()
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

#     bl_idname = "bdental4d.tresh_segment"
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


# class BDENTAL_4D_OT_MultiView(bpy.types.Operator):
#     """ MultiView Toggle """

#     bl_idname = "bdental4d.multiview"
#     bl_label = "MULTI-VIEW"

#     def execute(self, context):

#         Active_Obj = bpy.context.view_layer.objects.active

#         if not Active_Obj:
#             message = [" Please select CTVOLUME or SEGMENTATION ! "]
#             ShowMessageBox(message=message, icon="COLORSET_02_VEC")
#             return {"CANCELLED"}
#         else:
#             N = Active_Obj.name
#             Condition = [CheckString (N, ["BD4D", "_CTVolume"]) or \
#                         CheckString (N, ["BD4D", "SEGMENTATION"]) or \
#                         CheckString (N, ["BD4D", "_SLICES_POINTER"])] and \
#                         Active_Obj.select_get()

#             if not Condition:
#                 message = [
#                     " Please select CTVOLUME or SEGMENTATION or _SLICES_POINTER ! "
#                 ]
#                 ShowMessageBox(message=message, icon="COLORSET_02_VEC")
#                 return {"CANCELLED"}
#             else:
#                 Preffix = N.split('_')[0]
#                 AxialPlane = bpy.data.objects.get(f"1_{Preffix}_AXIAL_SLICE")
#                 CoronalPlane = bpy.data.objects.get(f"2_{Preffix}_CORONAL_SLICE")
#                 SagitalPlane = bpy.data.objects.get(f"3_{Preffix}_SAGITAL_SLICE")
#                 SLICES_POINTER = bpy.data.objects.get(f"{Preffix}_SLICES_POINTER")

#                 if not AxialPlane or not CoronalPlane or not SagitalPlane:
#                     message = [
#                         "To Add Multi-View Window :",
#                         "1 - Please select CTVOLUME or SEGMENTATION",
#                         "2 - Click on < SLICE VOLUME > button",
#                         "AXIAL, CORONAL and SAGITAL slices will be added",
#                         "3 - Click <MULTI-VIEW> button",
#                     ]
#                     ShowMessageBox(message=message, icon="COLORSET_02_VEC")
#                     return {"CANCELLED"}

#                 else:

#                     bpy.context.scene.unit_settings.scale_length = 0.001
#                     bpy.context.scene.unit_settings.length_unit = "MILLIMETERS"

#                     (
#                         MultiView_Window,
#                         OUTLINER,
#                         PROPERTIES,
#                         AXIAL,
#                         CORONAL,
#                         SAGITAL,
#                         VIEW_3D,
#                     ) = BDENTAL_4D_MultiView_Toggle(Preffix)
#                     MultiView_Screen = MultiView_Window.screen
#                     AXIAL_Space3D = [
#                         Space for Space in AXIAL.spaces if Space.type == "VIEW_3D"
#                     ][0]
#                     AXIAL_Region = [
#                         reg for reg in AXIAL.regions if reg.type == "WINDOW"
#                     ][0]

#                     CORONAL_Space3D = [
#                         Space for Space in CORONAL.spaces if Space.type == "VIEW_3D"
#                     ][0]
#                     CORONAL_Region = [
#                         reg for reg in CORONAL.regions if reg.type == "WINDOW"
#                     ][0]

#                     SAGITAL_Space3D = [
#                         Space for Space in SAGITAL.spaces if Space.type == "VIEW_3D"
#                     ][0]
#                     SAGITAL_Region = [
#                         reg for reg in SAGITAL.regions if reg.type == "WINDOW"
#                     ][0]
#                     # AXIAL Cam view toggle :

#                     AxialCam = bpy.data.objects.get(f"{AxialPlane.name}_CAM")
#                     AXIAL_Space3D.use_local_collections = True
#                     AXIAL_Space3D.use_local_camera = True
#                     AXIAL_Space3D.camera = AxialCam
#                     Override = {
#                         "window": MultiView_Window,
#                         "screen": MultiView_Screen,
#                         "area": AXIAL,
#                         "space_data": AXIAL_Space3D,
#                         "region": AXIAL_Region,
#                     }
#                     bpy.ops.view3d.view_camera(Override)

#                     # CORONAL Cam view toggle :
#                     CoronalCam = bpy.data.objects.get(f"{CoronalPlane.name}_CAM")
#                     CORONAL_Space3D.use_local_collections = True
#                     CORONAL_Space3D.use_local_camera = True
#                     CORONAL_Space3D.camera = CoronalCam
#                     Override = {
#                         "window": MultiView_Window,
#                         "screen": MultiView_Screen,
#                         "area": CORONAL,
#                         "space_data": CORONAL_Space3D,
#                         "region": CORONAL_Region,
#                     }
#                     bpy.ops.view3d.view_camera(Override)

#                     # AXIAL Cam view toggle :
#                     SagitalCam = bpy.data.objects.get(f"{SagitalPlane.name}_CAM")
#                     SAGITAL_Space3D.use_local_collections = True
#                     SAGITAL_Space3D.use_local_camera = True
#                     SAGITAL_Space3D.camera = SagitalCam
#                     Override = {
#                         "window": MultiView_Window,
#                         "screen": MultiView_Screen,
#                         "area": SAGITAL,
#                         "space_data": SAGITAL_Space3D,
#                         "region": SAGITAL_Region,
#                     }
#                     bpy.ops.view3d.view_camera(Override)

#                     bpy.ops.object.select_all(Override, action="DESELECT")
#                     SLICES_POINTER.select_set(True)
#                     bpy.context.view_layer.objects.active = SLICES_POINTER

#         return {"FINISHED"}


#######################################################################################
########################### Measurements : Operators ##############################
#######################################################################################


class BDENTAL_4D_OT_AddReferencePlanes(bpy.types.Operator):
    """ Add Reference Planes"""

    bl_idname = "bdental4d.add_reference_planes"
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
                    P = AddMarkupPoint(P_Names[0], self.Color, loc, 1, self.CollName)
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
            N = Active_Obj.name
            Condition = CheckString (N, ["BD4D"] ) and \
                        CheckString (N, ["_CTVolume", "SEGMENTATION"], any) and \
                        Active_Obj.select_get()
            
            if not Condition :
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
                    self.Preffix = self.TargetObject.name.split('_')[0]
                    DcmInfo = self.BDENTAL_4D_Props.DcmInfo
                    self.DcmInfo = eval(DcmInfo)
                    Override, area3D, space3D = CtxOverride(context)
                    # bpy.ops.screen.region_toggle(Override, region_type="UI")
                    bpy.ops.object.select_all(action="DESELECT")
                    # bpy.ops.object.select_all(Override, action="DESELECT")

                    context.window_manager.modal_handler_add(self)
                    self.BDENTAL_4D_Props.ActiveOperator = (
                        "bdental4d.add_reference_planes"
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

    bl_idname = "bdental4d.add_markup_point"
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
    Markup_Diameter: FloatProperty(
        description="Diameter", default=1, step=1, precision=2
    )

    CollName = "Markup Points"

    def execute(self, context):

        bpy.ops.object.mode_set(mode="OBJECT")

        if self.MarkupVoxelMode:
            Preffix = self.TargetObject.name.split('_')[0]
            CursorToVoxelPoint(Preffix=Preffix, CursorMove=True)

        Co = context.scene.cursor.location
        P = AddMarkupPoint(
            name=self.MarkupName,
            color=self.MarkupColor,
            loc=Co,
            Diameter=self.Markup_Diameter,
            CollName=self.CollName,
        )
        bpy.ops.object.select_all(action="DESELECT")
        self.TargetObject.select_set(True)
        bpy.context.view_layer.objects.active = self.TargetObject
        bpy.ops.object.mode_set(mode=self.mode)

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
                self.mode = Active_Obj.mode
                self.TargetObject = Active_Obj
                self.MarkupVoxelMode = CheckString (self.TargetObject.name, ["BD4D", "_CTVolume"] ) 
                wm = context.window_manager
                return wm.invoke_props_dialog(self)


class BDENTAL_4D_OT_CtVolumeOrientation(bpy.types.Operator):
    """ CtVolume Orientation according to Frankfort Plane """

    bl_idname = "bdental4d.ctvolume_orientation"
    bl_label = "CTVolume Orientation"

    def execute(self, context):

        BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props
        Active_Obj = bpy.context.view_layer.objects.active

        if not Active_Obj:
            message = [" Please select CTVOLUME for segmentation ! "]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")
            return {"CANCELLED"}

        else:

            Condition = CheckString (Active_Obj.name, ["BD4D"]) and  CheckString (Active_Obj.name, ["_CTVolume", 'Segmentation'], any ) 
            
            if not Condition:
                message = [
                    "CTVOLUME Orientation : ",
                    "Please select CTVOLUME or Segmentation! ",
                ]
                ShowMessageBox(message=message, icon="COLORSET_02_VEC")
                return {"CANCELLED"}

            else:
                Preffix = Active_Obj.name.split('_')[0]
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

    bl_idname = "bdental4d.reset_ctvolume_position"
    bl_label = "RESET CTVolume POSITION"

    def execute(self, context):

        BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props
        Active_Obj = bpy.context.view_layer.objects.active

        if not Active_Obj:
            message = [" Please select CTVOLUME for segmentation ! "]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")
            return {"CANCELLED"}
        else:
            Condition = CheckString (Active_Obj.name, ["BD4D"]) and  CheckString (Active_Obj.name, ["_CTVolume", 'Segmentation'], any ) 


            if not Condition :
                message = [
                    "Reset Position : ",
                    "Please select CTVOLUME or Segmentation! ",
                ]
                ShowMessageBox(message=message, icon="COLORSET_02_VEC")
                return {"CANCELLED"}

            else:
                Preffix = Active_Obj.name.split('_')[0]
                Vol = [
                    obj
                    for obj in bpy.data.objects
                    if CheckString (obj.name, [Preffix, "_CTVolume"])
                ][0]
                DcmInfoDict = eval(BDENTAL_4D_Props.DcmInfo)
                DcmInfo = DcmInfoDict[Preffix]
                TransformMatrix = DcmInfo["TransformMatrix"]
                Vol.matrix_world = TransformMatrix

                return {"FINISHED"}


class BDENTAL_4D_OT_AddTeeth(bpy.types.Operator):
    """ Add Teeth """

    bl_idname = "bdental4d.add_teeth"
    bl_label = "ADD TEETH"
    bl_options = {"REGISTER", "UNDO"}

    def modal(self, context, event):

        ############################################
        if not event.type in {
            "RET",
            "ESC",
        }:
            # allow navigation

            return {"PASS_THROUGH"}

        ###########################################
        elif event.type == "RET":

            if event.value == ("PRESS"):

                Override, area3D, space3D = CtxOverride(context)

                Selected_Teeth = context.selected_objects
                bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")

                for obj in self.Coll.objects:
                    if not obj in Selected_Teeth:
                        bpy.data.objects.remove(obj)

                # Restore scene :
                bpy.context.space_data.shading.background_color = self.ViewPortColor
                bpy.context.space_data.shading.color_type = self.ColorType
                bpy.context.space_data.shading.background_type = self.SolidType
                bpy.context.space_data.shading.type = self.BackGroundType

                space3D.overlay.show_annotation = True
                space3D.overlay.show_extras = True
                space3D.overlay.show_floor = True
                space3D.overlay.show_axis_x = True
                space3D.overlay.show_axis_y = True

                if self.visibleObjects:
                    for Name in self.visibleObjects:
                        obj = bpy.data.objects.get(Name)
                        if obj:
                            obj.hide_set(False)

                bpy.ops.object.select_all(Override, action="DESELECT")
                bpy.ops.screen.screen_full_area(Override)

                return {"FINISHED"}

        ###########################################
        elif event.type == ("ESC"):

            if event.value == ("PRESS"):
                Override, area3D, space3D = CtxOverride(context)
                # Restore scene :
                bpy.context.space_data.shading.background_color = self.ViewPortColor
                bpy.context.space_data.shading.color_type = self.ColorType
                bpy.context.space_data.shading.background_type = self.SolidType
                bpy.context.space_data.shading.type = self.BackGroundType

                space3D.overlay.show_annotation = True
                space3D.overlay.show_extras = True
                space3D.overlay.show_floor = True
                space3D.overlay.show_axis_x = True
                space3D.overlay.show_axis_y = True

                if self.visibleObjects:
                    for Name in self.visibleObjects:
                        obj = bpy.data.objects.get(Name)
                        if obj:
                            obj.hide_set(False)

                for obj in self.Coll.objects:
                    bpy.data.objects.remove(obj)

                bpy.data.collections.remove(self.Coll)
                bpy.ops.object.select_all(Override, action="DESELECT")
                bpy.ops.screen.screen_full_area(Override)

                message = [
                    " Add Teeth Operation was Cancelled!",
                ]

                ShowMessageBox(message=message, icon="COLORSET_03_VEC")

                return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):

        if context.space_data.type == "VIEW_3D":

            BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props

            bpy.ops.screen.screen_full_area()
            Override, area3D, space3D = CtxOverride(context)
            bpy.ops.object.select_all(Override, action="DESELECT")

            ###########################################################
            self.TeethLibrary = BDENTAL_4D_Props.TeethLibrary

            self.visibleObjects = [obj.name for obj in bpy.context.visible_objects]

            self.BackGroundType = space3D.shading.type
            space3D.shading.type == "SOLID"

            self.SolidType = space3D.shading.background_type
            space3D.shading.background_type = "VIEWPORT"

            self.ColorType = space3D.shading.color_type
            space3D.shading.color_type = "MATERIAL"

            self.ViewPortColor = tuple(space3D.shading.background_color)
            space3D.shading.background_color = (0.0, 0.0, 0.0)

            # Prepare scene  :
            ##########################################################

            space3D.overlay.show_outline_selected = True
            space3D.overlay.show_object_origins = True
            space3D.overlay.show_annotation = False
            space3D.overlay.show_text = True
            space3D.overlay.show_extras = False
            space3D.overlay.show_floor = False
            space3D.overlay.show_axis_x = False
            space3D.overlay.show_axis_y = False

            for Name in self.visibleObjects:
                obj = bpy.data.objects.get(Name)
                if obj:
                    obj.hide_set(True)

            filename = self.TeethLibrary
            directory = join(DataBlendFile, "Collection")
            bpy.ops.wm.append(directory=directory, filename=filename)
            Coll = bpy.data.collections.get(self.TeethLibrary)

            for obj in context.selected_objects:
                MoveToCollection(obj=obj, CollName="Teeth")
            bpy.data.collections.remove(Coll)

            self.Coll = bpy.data.collections.get("Teeth")

            bpy.ops.object.select_all(Override, action="DESELECT")

            context.window_manager.modal_handler_add(self)

            return {"RUNNING_MODAL"}

        else:

            message = [
                "Active space must be a View3d",
            ]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}


class BDENTAL_4D_OT_AddImplantSleeve(bpy.types.Operator):
    """ Add Sleeve """

    bl_idname = "bdental4d.add_implant_sleeve"
    bl_label = "IMPLANT SLEEVE"

    def execute(self, context):
        if not context.active_object:
            message = ["Please select The Implant!"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        if not context.active_object.select_get() or not "IMPLANT" in context.active_object.name :
            message = message = ["Please select the Implant!"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:
            Implant = context.active_object
            cursor = bpy.context.scene.cursor
            cursor.matrix = Implant.matrix_world
            bpy.ops.bdental4d.add_sleeve(Orientation="AXIAL")

            Sleeve = context.active_object
            Implant.select_set(True)
            context.view_layer.objects.active = Implant
            bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)

            bpy.ops.object.select_all(action="DESELECT")
            Sleeve.select_set(True)
            context.view_layer.objects.active = Sleeve

            return {"FINISHED"}


class BDENTAL_4D_OT_AddSleeve(bpy.types.Operator):
    """ Add Sleeve """

    bl_idname = "bdental4d.add_sleeve"
    bl_label = "ADD SLEEVE"

    OrientationTypes = ["AXIAL", "SAGITAL/CORONAL"]
    items = []
    for i in range(len(OrientationTypes)):
        item = (str(OrientationTypes[i]), str(OrientationTypes[i]), str(""), int(i))
        items.append(item)

    Orientation: EnumProperty(items=items, description="Orientation", default="AXIAL")

    def execute(self, context):

        BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props
        self.SleeveDiametre = BDENTAL_4D_Props.SleeveDiameter
        self.SleeveHeight = BDENTAL_4D_Props.SleeveHeight
        self.HoleDiameter = BDENTAL_4D_Props.HoleDiameter
        self.HoleOffset = BDENTAL_4D_Props.HoleOffset
        self.cursor = context.scene.cursor

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

        for obj in [Pin, Sleeve]:
            MoveToCollection(obj, "GUIDE Components")

        return {"FINISHED"}

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)


class BDENTAL_4D_OT_AddImplant(bpy.types.Operator):
    """ Add Implant """

    bl_idname = "bdental4d.add_implant"
    bl_label = "ADD IMPLANT"

    OrientationTypes = ["AXIAL", "SAGITAL/CORONAL"]
    items = []
    for i in range(len(OrientationTypes)):
        item = (str(OrientationTypes[i]), str(OrientationTypes[i]), str(""), int(i))
        items.append(item)

    # Orientation: EnumProperty(items=items, description="Orientation", default="AXIAL")

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

    def AddImplant(self, context):
        ImplantLibraryBlendFile = join(
            addon_dir, "Resources", "BlendData", "BDENTAL4D_Implants_Library.blend"
        )
        filename = f"{self.Implant_Diameter}_{self.Implant_Lenght}_IMPLANT_({self.ImplantLibrary})"
        directory = join(ImplantLibraryBlendFile, "Object")
        
        bpy.ops.wm.append(directory=directory, filename=filename)
        self.Implant = bpy.data.objects.get(filename)
        context.view_layer.objects.active = self.Implant

    def execute(self, context):

        self.AddImplant(context)
        self.Implant.matrix_world = self.matrix
        MoveToCollection(self.Implant, "GUIDE Components")
        self.Pointer.select_set(True)
        context.view_layer.objects.active = self.Pointer
        bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)
        bpy.ops.object.select_all(action="DESELECT")
        self.Pointer.select_set(True)
        context.view_layer.objects.active = self.Pointer

        return {"FINISHED"}

    def invoke(self, context, event):
        BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props
        if not context.active_object:
            message = ["Please select The SLICES_POINTER!"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        if not context.active_object.select_get() or not context.object.name.endswith(
            "_SLICES_POINTER"
        ):
            message = message = ["Please select the SLICES_POINTER!"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:
            self.matrix = context.object.matrix_world
            self.Pointer = context.active_object

            self.ImplantLibrary = BDENTAL_4D_Props.ImplantLibrary
            wm = context.window_manager
            return wm.invoke_props_dialog(self)

class BDENTAL_4D_OT_AlignImplants(bpy.types.Operator):
    """ Align Implants """

    bl_idname = "bdental4d.align_implants"
    bl_label = "ALIGN IMPLANTS AXES"

    AlignModes = ["To Active", "Averrage Axes"]
    items = []
    for i in range(len(AlignModes)):
        item = (str(AlignModes[i]), str(AlignModes[i]), str(""), int(i))
        items.append(item)

    AlignMode: EnumProperty(items=items, description="Implant Align Mode", default="To Active")


    def execute(self, context):
        if self.AlignMode == "Averrage Axes" :
            MeanRot = np.mean([ np.array(Impt.rotation_euler) for Impt in self.Implants], axis=0)
            for Impt in self.Implants :
                Impt.rotation_euler = MeanRot
            
        elif self.AlignMode == "To Active" :
            for Impt in self.Implants :
                Impt.rotation_euler = self.Active_Imp.rotation_euler
        return {"FINISHED"}

    def invoke(self, context, event):
        
        if not context.active_object:
            message = ["Please select 2 implants at least"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")
            return {"CANCELLED"}

        if not context.active_object.select_get() or not len(context.selected_objects) >=2 :
            message = message = ["Please select 2 implants at least"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:
            self.Active_Imp = context.active_object
            self.Implants = context.selected_objects
            
            wm = context.window_manager
            return wm.invoke_props_dialog(self)


##################################################################
class BDENTAL_4D_OT_AddSplint(bpy.types.Operator):
    """ Add Splint """

    bl_idname = "bdental4d.add_splint"
    bl_label = "Splint"
    bl_options = {"REGISTER", "UNDO"}

    thikness: FloatProperty(
        description="SPLINT thikness", default=2, step=1, precision=2
    )

    def execute(self, context):

        Splint = Metaball_Splint(self.BaseMesh, self.thikness)

        return {"FINISHED"}

    def invoke(self, context, event):

        if not context.object:
            message = ["Please select a base mesh!"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        if not context.object.select_get() or context.object.type != "MESH":
            message = ["Please select a base mesh!"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:
            self.BaseMesh = context.object
            wm = context.window_manager
            return wm.invoke_props_dialog(self)


class BDENTAL_4D_OT_Survey(bpy.types.Operator):
    " Survey the model from view top"

    bl_idname = "bdental4d.survey"
    bl_label = "Survey Model"

    SurveyColor: FloatVectorProperty(
        name="Survey Color",
        description="Survey Color",
        default=[0.2, 0.12, 0.17, 1.0],
        soft_min=0.0,
        soft_max=1.0,
        size=4,
        subtype="COLOR",
    )

    def execute(self, context):
        BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props
        bpy.ops.object.mode_set(mode="OBJECT")
        Old_Survey_mat = bpy.data.materials.get("BDENTAL_4D_survey_mat")
        if Old_Survey_mat:
            OldmatSlotsIds = [
                i
                for i in range(len(self.Model.material_slots))
                if self.Model.material_slots[i].material == Old_Survey_mat
            ]
            if OldmatSlotsIds:
                for idx in OldmatSlotsIds:
                    self.Model.active_material_index = idx
                    bpy.ops.object.material_slot_remove()

        Override, area3D, space3D = CtxOverride(context)
        view_mtx = space3D.region_3d.view_matrix.copy()
        if not self.Model.data.materials[:]:
            ModelMat = bpy.data.materials.get(
                "BDENTAL_4D_Neutral_mat"
            ) or bpy.data.materials.new("BDENTAL_4D_Neutral_mat")
            ModelMat.diffuse_color = (0.8, 0.8, 0.8, 1.0)
            self.Model.active_material = ModelMat

        Survey_mat = bpy.data.materials.get(
            "BDENTAL_4D_survey_mat"
        ) or bpy.data.materials.new("BDENTAL_4D_survey_mat")
        Survey_mat.diffuse_color = self.SurveyColor
        self.Model.data.materials.append(Survey_mat)
        self.Model.active_material_index = len(self.Model.material_slots) - 1

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")

        # #############################____Surveying____###############################
        survey_faces_index_list = []

        obj = self.Model
        View_Local_Z = obj.matrix_world.inverted().to_quaternion() @ (
            space3D.region_3d.view_rotation @ Vector((0, 0, 1))
        )

        survey_faces_Idx = [
            f.index for f in obj.data.polygons if f.normal.dot(View_Local_Z) < -0.000001
        ]

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (False, False, True)
        bpy.ops.mesh.select_all(action="DESELECT")

        bpy.ops.object.mode_set(mode="OBJECT")

        for i in survey_faces_Idx:
            f = obj.data.polygons[i]
            f.select = True

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        Survey_Vg = obj.vertex_groups.get(
            "BDENTAL_4D_survey_vg"
        ) or obj.vertex_groups.new(name="BDENTAL_4D_survey_vg")
        # obj.vertex_groups.active_index = Survey_Vg.index
        bpy.ops.object.vertex_group_assign()
        bpy.ops.object.material_slot_assign()
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)

        # Store Survey direction :
        SurveyInfo_Dict = eval(BDENTAL_4D_Props.SurveyInfo)
        SurveyInfo_Dict[obj.as_pointer()] = (View_Local_Z, Survey_mat)
        BDENTAL_4D_Props.SurveyInfo = str(SurveyInfo_Dict)

        return {"FINISHED"}

    def invoke(self, context, event):

        if not context.active_object:
            message = ["Please select Model to survey!"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        if (
            not context.active_object.select_get()
            or context.active_object.type != "MESH"
        ):
            message = ["Please select Model to survey!"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:
            self.Model = context.active_object
            wm = context.window_manager
            return wm.invoke_props_dialog(self)


class BDENTAL_4D_OT_ModelBase(bpy.types.Operator):
    """Make a model base from top user view prspective"""

    bl_idname = "bdental4d.model_base"
    bl_label = "Model Base"
    bl_options = {"REGISTER", "UNDO"}

    Model_Types = ["Upper Model", "Lower Model"]
    items = []
    for i in range(len(Model_Types)):
        item = (str(Model_Types[i]), str(Model_Types[i]), str(""), int(i))
        items.append(item)

    ModelType: EnumProperty(
        items=items, description="Model Type", default="Upper Model"
    )
    HollowModel: BoolProperty(
        name="Make Hollow Model",
        description="Add Hollow Model",
        default=False,
    )

    def execute(self, context):
        if not context.active_object:
            message = ["Please select target mesh !"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        if (
            not context.active_object.select_get()
            or context.active_object.type != "MESH"
        ):
            message = ["Please select target mesh !"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:
            # Check base boarder :
            TargetMesh = context.active_object
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.context.tool_settings.mesh_select_mode = (True, False, False)
            bpy.ops.mesh.select_all(action="DESELECT")
            bpy.ops.mesh.select_non_manifold()
            bpy.ops.object.mode_set(mode="OBJECT")
            NonManifoldVerts = [v for v in TargetMesh.data.vertices if v.select]

            if not NonManifoldVerts:
                message = [
                    "The target mesh is closed !",
                    "Can't make base from Closed mesh.",
                ]
                ShowMessageBox(message=message, icon="COLORSET_02_VEC")

                return {"CANCELLED"}

            else:
                BaseHeight = context.scene.BDENTAL_4D_Props.BaseHeight
                obj = TargetMesh

                ####### Duplicate Target Mesh #######
                bpy.ops.object.select_all(action="DESELECT")
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.duplicate_move()

                ModelBase = context.object
                ModelBase.name = f"{TargetMesh.name} (BASE MODEL)"
                ModelBase.data.name = ModelBase.name
                obj = ModelBase
                # Relax border loop :
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_all(action="DESELECT")
                bpy.ops.mesh.select_non_manifold()
                bpy.ops.mesh.remove_doubles(threshold=0.1)
                bpy.ops.mesh.looptools_relax(
                    input="selected",
                    interpolation="cubic",
                    iterations="3",
                    regular=True,
                )

                # Make some calcul of average z_cordinate of border vertices :

                bpy.ops.object.mode_set(mode="OBJECT")

                obj_mx = obj.matrix_world.copy()
                verts = obj.data.vertices
                global_z_cords = [(obj_mx @ v.co)[2] for v in verts]

                HollowOffset = 0
                if self.ModelType == "Upper Model":
                    Extrem_z = max(global_z_cords)
                    Delta = BaseHeight
                    if self.HollowModel:
                        HollowOffset = 4
                        BisectPlaneLoc = Vector((0, 0, Extrem_z))
                        BisectPlaneNormal = Vector((0, 0, 1))

                if self.ModelType == "Lower Model":
                    Extrem_z = min(global_z_cords)
                    Delta = -BaseHeight
                    if self.HollowModel:
                        HollowOffset = -4
                        BisectPlaneLoc = Vector((0, 0, Extrem_z))
                        BisectPlaneNormal = Vector((0, 0, -1))

                # Border_2 = Extrude 1st border loop no translation :
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.extrude_region_move()

                # change Border2 vertices zco to min_z - base_height  :

                bpy.ops.object.mode_set(mode="OBJECT")
                selected_verts = [v for v in verts if v.select == True]

                for v in selected_verts:
                    global_v_co = obj_mx @ v.co
                    v.co = obj_mx.inverted() @ Vector(
                        (
                            global_v_co[0],
                            global_v_co[1],
                            Extrem_z + Delta + HollowOffset,
                        )
                    )

                # fill base :
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.edge_face_add()
                bpy.ops.mesh.dissolve_limited()

                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.fill_holes(sides=100)

                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.normals_make_consistent(inside=False)
                bpy.ops.mesh.select_all(action="DESELECT")
                bpy.ops.object.mode_set(mode="OBJECT")

                if self.HollowModel:
                    bpy.ops.bdental4d.hollow_model(thikness=2)
                    HollowModel = context.active_object

                    bpy.ops.object.mode_set(mode="EDIT")
                    bpy.ops.mesh.select_all(action="SELECT")

                    bpy.ops.mesh.bisect(
                        plane_co=BisectPlaneLoc,
                        plane_no=BisectPlaneNormal,
                        use_fill=True,
                        clear_inner=False,
                        clear_outer=True,
                    )
                    bpy.ops.mesh.select_all(action="DESELECT")
                    bpy.ops.object.mode_set(mode="OBJECT")

                    bpy.ops.object.select_all(action="DESELECT")
                    obj.select_set(True)
                    bpy.context.view_layer.objects.active = obj
                    bpy.ops.object.mode_set(mode="EDIT")
                    bpy.ops.mesh.select_all(action="SELECT")
                    bpy.ops.mesh.bisect(
                        plane_co=BisectPlaneLoc,
                        plane_no=BisectPlaneNormal,
                        use_fill=True,
                        clear_inner=False,
                        clear_outer=True,
                    )
                    bpy.ops.mesh.select_all(action="DESELECT")
                    bpy.ops.object.mode_set(mode="OBJECT")

                message = ["Model Base created successfully"]
                ShowMessageBox(message=message, icon="COLORSET_03_VEC")

                return {"FINISHED"}

    def invoke(self, context, event):

        Active_Obj = context.active_object

        if not Active_Obj:
            message = [" Please select Target mesh Object ! "]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")
            return {"CANCELLED"}

        if Active_Obj.select_get() == False or Active_Obj.type != "MESH":
            message = [" Please select Target mesh Object ! "]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")
            return {"CANCELLED"}

        else:
            self.Active_Obj = Active_Obj
            wm = context.window_manager
            return wm.invoke_props_dialog(self)


class BDENTAL_4D_OT_hollow_model(bpy.types.Operator):
    """Create a hollow Dental Model from closed Model """

    bl_idname = "bdental4d.hollow_model"
    bl_label = "Hollow Model"
    bl_options = {"REGISTER", "UNDO"}

    thikness: FloatProperty(description="OFFSET", default=2, step=1, precision=2)

    def execute(self, context):

        Model = context.active_object
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_non_manifold()

        bpy.ops.object.mode_set(mode="OBJECT")
        verts = Model.data.vertices
        selected_verts = [v for v in verts if v.select]

        if selected_verts:

            message = [" Invalid mesh! Please clean mesh first !"]
            ShowMessageBox(message=message, icon="COLORSET_01_VEC")

            return {"CANCELLED"}

        else:

            # Prepare scene settings :

            bpy.ops.view3d.snap_cursor_to_center()
            bpy.ops.transform.select_orientation(orientation="GLOBAL")
            bpy.context.scene.tool_settings.transform_pivot_point = "INDIVIDUAL_ORIGINS"
            bpy.context.tool_settings.mesh_select_mode = (True, False, False)
            bpy.context.scene.tool_settings.use_snap = False
            bpy.context.scene.tool_settings.use_proportional_edit_objects = False
            bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")

            ####### Duplicate Model #######

            # Duplicate Model to Model_hollow:

            bpy.ops.object.select_all(action="DESELECT")
            Model.select_set(True)
            bpy.context.view_layer.objects.active = Model
            bpy.ops.object.duplicate_move()

            # Rename Model_hollow....

            Model_hollow = context.active_object
            Model_hollow.name = Model.name + "_hollow"

            # Duplicate Model_hollow and make a low resolution duplicate :

            bpy.ops.object.duplicate_move()

            # Rename Model_lowres :

            Model_lowres = bpy.context.view_layer.objects.active
            Model_lowres.name = "Model_lowres"
            mesh_lowres = Model_lowres.data
            mesh_lowres.name = "Model_lowres_mesh"

            # Get Model_lowres :

            bpy.ops.object.select_all(action="DESELECT")
            Model_lowres.select_set(True)
            bpy.context.view_layer.objects.active = Model_lowres

            # remesh Model_lowres 1.0 mm :

            bpy.context.object.data.use_remesh_smooth_normals = True
            bpy.context.object.data.use_remesh_preserve_volume = True
            bpy.context.object.data.use_remesh_fix_poles = True
            bpy.context.object.data.remesh_voxel_size = 1
            bpy.ops.object.voxel_remesh()

            # Add Metaballs :

            obj = Model_lowres

            loc, rot, scale = obj.matrix_world.decompose()

            verts = obj.data.vertices
            vcords = [rot @ v.co + loc for v in verts]
            mball_elements_cords = [vco - vcords[0] for vco in vcords[1:]]

            bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.select_all(action="DESELECT")

            thikness = self.thikness
            radius = thikness * 5 / 8

            bpy.ops.object.metaball_add(
                type="BALL", radius=radius, enter_editmode=False, location=vcords[0]
            )

            Mball_object = bpy.context.view_layer.objects.active
            Mball_object.name = "Mball_object"
            mball = Mball_object.data
            mball.resolution = 0.6
            bpy.context.object.data.update_method = "FAST"

            for i in range(len(mball_elements_cords)):
                element = mball.elements.new()
                element.co = mball_elements_cords[i]
                element.radius = radius * 2

            bpy.ops.object.convert(target="MESH")

            Mball_object = bpy.context.view_layer.objects.active
            Mball_object.name = "Mball_object"
            mball_mesh = Mball_object.data
            mball_mesh.name = "Mball_object_mesh"

            # Get Hollow Model :

            bpy.ops.object.select_all(action="DESELECT")
            Model_hollow.select_set(True)
            bpy.context.view_layer.objects.active = Model_hollow

            # Make boolean intersect operation :
            bpy.ops.object.modifier_add(type="BOOLEAN")
            bpy.context.object.modifiers["Boolean"].show_viewport = False
            bpy.context.object.modifiers["Boolean"].operation = "INTERSECT"
            bpy.context.object.modifiers["Boolean"].object = Mball_object

            bpy.ops.object.modifier_apply(modifier="Boolean")

            # Delet Model_lowres and Mball_object:
            bpy.data.objects.remove(Model_lowres)
            bpy.data.objects.remove(Mball_object)

            # Hide everything but hollow model + Model :

            bpy.ops.object.select_all(action="DESELECT")

            return {"FINISHED"}

    def invoke(self, context, event):

        Active_Obj = context.active_object

        if not Active_Obj:
            message = [" Please select Target mesh Object ! "]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")
            return {"CANCELLED"}

        if Active_Obj.select_get() == False or Active_Obj.type != "MESH":
            message = [" Please select Target mesh Object ! "]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")
            return {"CANCELLED"}

        else:
            wm = context.window_manager
            return wm.invoke_props_dialog(self)


class BDENTAL_4D_OT_BlockModel(bpy.types.Operator):
    " Blockout Model (Remove Undercuts)"

    bl_idname = "bdental4d.block_model"
    bl_label = "BLOCK Model"

    def execute(self, context):

        if not context.active_object:
            message = ["Please select Model to Blockout !"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        if (
            not context.active_object.select_get()
            or context.active_object.type != "MESH"
        ):
            message = ["Please select Model to Blockout !"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:
            Model = context.active_object
            Pointer = Model.as_pointer()
            BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props
            SurveyInfo_Dict = eval(BDENTAL_4D_Props.SurveyInfo)
            if not Pointer in SurveyInfo_Dict.keys():
                message = ["Please Survey Model before Blockout !"]
                ShowMessageBox(message=message, icon="COLORSET_02_VEC")

                return {"CANCELLED"}
            else:
                View_Local_Z, Survey_mat = SurveyInfo_Dict[Pointer]
                ExtrudeVector = -20 * (
                    Model.matrix_world.to_quaternion() @ View_Local_Z
                )

                print(ExtrudeVector)

                # duplicate Model :
                bpy.ops.object.select_all(action="DESELECT")
                Model.select_set(True)
                bpy.context.view_layer.objects.active = Model

                bpy.ops.object.duplicate_move()
                BlockedModel = bpy.context.view_layer.objects.active
                BlockedModel.name = f"{Model.name}(BLOCKED)"
                BlockedModel.data.name = BlockedModel.name

                for _ in BlockedModel.material_slots:
                    bpy.ops.object.material_slot_remove()

                BlockedModel.active_material = Survey_mat
                bpy.ops.object.mode_set(mode="EDIT")

                bpy.context.tool_settings.mesh_select_mode = (False, False, True)
                bpy.ops.mesh.extrude_region_move()
                bpy.ops.transform.translate(value=ExtrudeVector)

                bpy.ops.object.mode_set(mode="OBJECT")
                BlockedModel.data.remesh_mode = "VOXEL"
                BlockedModel.data.remesh_voxel_size = 0.2
                BlockedModel.data.use_remesh_fix_poles = True
                BlockedModel.data.use_remesh_smooth_normals = True
                BlockedModel.data.use_remesh_preserve_volume = True

                bpy.ops.object.voxel_remesh()

                return {"FINISHED"}


class BDENTAL_4D_OT_add_offset(bpy.types.Operator):
    """ Add offset to mesh """

    bl_idname = "bdental4d.add_offset"
    bl_label = "Add Offset"
    bl_options = {"REGISTER", "UNDO"}

    Offset: FloatProperty(description="OFFSET", default=0.1, step=1, precision=2)

    def execute(self, context):

        offset = round(self.Offset, 2)

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.modifier_add(type="DISPLACE")
        bpy.context.object.modifiers["Displace"].mid_level = 0
        bpy.context.object.modifiers["Displace"].strength = offset
        bpy.ops.object.modifier_apply(modifier="Displace")

        return {"FINISHED"}

    def invoke(self, context, event):

        Active_Obj = context.active_object

        if not Active_Obj:
            message = [" Please select Target mesh Object ! "]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")
            return {"CANCELLED"}

        if Active_Obj.select_get() == False or Active_Obj.type != "MESH":
            message = [" Please select Target mesh Object ! "]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")
            return {"CANCELLED"}

        else:
            wm = context.window_manager
            return wm.invoke_props_dialog(self)


#######################################################################
# Align Operators
#######################################################################

############################################################################
class BDENTAL_4D_OT_AlignPoints(bpy.types.Operator):
    """ Add Align Refference points """

    bl_idname = "bdental4d.alignpoints"
    bl_label = "ALIGN POINTS"
    bl_options = {"REGISTER", "UNDO"}

    TargetColor = (0, 1, 0, 1)  # Green
    SourceColor = (1, 0, 0, 1)  # Red
    CollName = "ALIGN POINTS"
    TargetChar = "B"
    SourceChar = "A"

    def IcpPipline(
        self,
        SourceObj,
        TargetObj,
        SourceVidList,
        TargetVidList,
        VertsLimite,
        Iterations,
        Precision,
    ):

        MaxDist = 0.0
        for i in range(Iterations):

            SourceVcoList = [
                SourceObj.matrix_world @ SourceObj.data.vertices[idx].co
                for idx in SourceVidList
            ]
            TargetVcoList = [
                TargetObj.matrix_world @ TargetObj.data.vertices[idx].co
                for idx in TargetVidList
            ]

            (
                SourceKdList,
                TargetKdList,
                DistList,
                SourceIndexList,
                TargetIndexList,
            ) = KdIcpPairs(SourceVcoList, TargetVcoList, VertsLimite=VertsLimite)

            TransformMatrix = KdIcpPairsToTransformMatrix(
                TargetKdList=TargetKdList, SourceKdList=SourceKdList
            )
            SourceObj.matrix_world = TransformMatrix @ SourceObj.matrix_world
            for RefP in self.SourceRefPoints:
                RefP.matrix_world = TransformMatrix @ RefP.matrix_world
            # Update scene :
            SourceObj.update_tag()
            bpy.context.view_layer.update()

            SourceObj = self.SourceObject

            SourceVcoList = [
                SourceObj.matrix_world @ SourceObj.data.vertices[idx].co
                for idx in SourceVidList
            ]
            _, _, DistList, _, _ = KdIcpPairs(
                SourceVcoList, TargetVcoList, VertsLimite=VertsLimite
            )
            MaxDist = max(DistList)
            Override, area3D, space3D = CtxOverride(bpy.context)
            bpy.ops.wm.redraw_timer(Override, type="DRAW_WIN_SWAP", iterations=1)
            #######################################################
            if MaxDist <= Precision:
                self.ResultMessage = [
                    "Allignement Done !",
                    f"Max Distance < or = {Precision} mm",
                ]
                print(f"Number of iterations = {i}")
                print(f"Precision of {Precision} mm reached.")
                print(f"Max Distance = {round(MaxDist, 6)} mm")
                break

        if MaxDist > Precision:
            print(f"Number of iterations = {i}")
            print(f"Max Distance = {round(MaxDist, 6)} mm")
            self.ResultMessage = [
                "Allignement Done !",
                f"Max Distance = {round(MaxDist, 6)} mm",
            ]

    def modal(self, context, event):

        ############################################
        if not event.type in {
            self.TargetChar,
            self.SourceChar,
            "DEL",
            "RET",
            "ESC",
        }:
            # allow navigation

            return {"PASS_THROUGH"}
        #########################################
        if event.type == self.TargetChar:
            # Add Target Refference point :
            if event.value == ("PRESS"):
                if self.TargetVoxelMode:
                    Preffix = self.TargetObject.name.split('_')[0]
                    CursorToVoxelPoint(Preffix=Preffix, CursorMove=True)

                color = self.TargetColor
                CollName = self.CollName
                self.TargetCounter += 1
                name = f"B{self.TargetCounter}"
                RefP = AddRefPoint(name, color, CollName)
                self.TargetRefPoints.append(RefP)
                self.TotalRefPoints.append(RefP)
                bpy.ops.object.select_all(action="DESELECT")

        #########################################
        if event.type == self.SourceChar:
            # Add Source Refference point :
            if event.value == ("PRESS"):
                if self.SourceVoxelMode:
                    Preffix = self.SourceObject.name.split('_')[0]
                    CursorToVoxelPoint(Preffix=Preffix, CursorMove=True)

                color = self.SourceColor
                CollName = self.CollName
                self.SourceCounter += 1
                name = f"M{self.SourceCounter}"
                RefP = AddRefPoint(name, color, CollName)
                self.SourceRefPoints.append(RefP)
                self.TotalRefPoints.append(RefP)
                bpy.ops.object.select_all(action="DESELECT")

        ###########################################
        elif event.type == ("DEL"):
            if event.value == ("PRESS"):
                if self.TotalRefPoints:
                    obj = self.TotalRefPoints.pop()
                    name = obj.name
                    if name.startswith("B"):
                        self.TargetCounter -= 1
                        self.TargetRefPoints.pop()
                    if name.startswith("M"):
                        self.SourceCounter -= 1
                        self.SourceRefPoints.pop()
                    bpy.data.objects.remove(obj)
                    bpy.ops.object.select_all(action="DESELECT")

        ###########################################
        elif event.type == "RET":

            if event.value == ("PRESS"):

                start = Tcounter()

                TargetObj = self.TargetObject
                SourceObj = self.SourceObject

                #############################################
                condition = (
                    len(self.TargetRefPoints) == len(self.SourceRefPoints)
                    and len(self.TargetRefPoints) >= 3
                )
                if not condition:
                    message = [
                        "          Please check the following :",
                        "   - The number of Base Refference points and,",
                        "       Align Refference points should match!",
                        "   - The number of Base Refference points ,",
                        "         and Align Refference points,",
                        "       should be superior or equal to 3",
                        "        <<Please check and retry !>>",
                    ]
                    ShowMessageBox(message=message, icon="COLORSET_02_VEC")

                else:

                    TransformMatrix = RefPointsToTransformMatrix(
                        self.TargetRefPoints, self.SourceRefPoints
                    )

                    SourceObj.matrix_world = TransformMatrix @ SourceObj.matrix_world
                    for SourceRefP in self.SourceRefPoints:
                        SourceRefP.matrix_world = (
                            TransformMatrix @ SourceRefP.matrix_world
                        )

                    for i, SP in enumerate(self.SourceRefPoints):
                        TP = self.TargetRefPoints[i]
                        MidLoc = (SP.location + TP.location) / 2
                        SP.location = TP.location = MidLoc

                    # Update scene :
                    context.view_layer.update()
                    for obj in [TargetObj, SourceObj]:
                        obj.update_tag()
                    bpy.ops.wm.redraw_timer(
                        self.FullOverride, type="DRAW_WIN_SWAP", iterations=1
                    )

                    self.ResultMessage = []
                    if not self.TargetVoxelMode and not self.SourceVoxelMode:
                        #########################################################
                        # ICP alignement :
                        print("ICP Align processing...")
                        IcpVidDict = VidDictFromPoints(
                            TargetRefPoints=self.TargetRefPoints,
                            SourceRefPoints=self.SourceRefPoints,
                            TargetObj=TargetObj,
                            SourceObj=SourceObj,
                            radius=3,
                        )
                        BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props
                        BDENTAL_4D_Props.IcpVidDict = str(IcpVidDict)

                        SourceVidList, TargetVidList = (
                            IcpVidDict[SourceObj],
                            IcpVidDict[TargetObj],
                        )

                        self.IcpPipline(
                            SourceObj=SourceObj,
                            TargetObj=TargetObj,
                            SourceVidList=SourceVidList,
                            TargetVidList=TargetVidList,
                            VertsLimite=10000,
                            Iterations=30,
                            Precision=0.0001,
                        )

                    ##########################################################
                    self.FullSpace3D.overlay.show_outline_selected = True
                    self.FullSpace3D.overlay.show_object_origins = True
                    self.FullSpace3D.overlay.show_annotation = True
                    self.FullSpace3D.overlay.show_text = True
                    self.FullSpace3D.overlay.show_extras = True
                    self.FullSpace3D.overlay.show_floor = True
                    self.FullSpace3D.overlay.show_axis_x = True
                    self.FullSpace3D.overlay.show_axis_y = True
                    ###########################################################
                    for Name in self.visibleObjects:
                        obj = bpy.data.objects.get(Name)
                        if obj:
                            obj.hide_set(False)

                    bpy.ops.object.select_all(self.FullOverride, action="DESELECT")
                    bpy.ops.wm.tool_set_by_id(self.FullOverride, name="builtin.select")
                    bpy.context.scene.tool_settings.use_snap = False
                    bpy.context.scene.cursor.location = (0, 0, 0)
                    bpy.ops.screen.region_toggle(self.FullOverride, region_type="UI")

                    if self.Solid:
                        self.FullSpace3D.shading.background_color = (
                            self.background_color
                        )
                        self.FullSpace3D.shading.background_type = self.background_type

                    TargetObj = self.TargetObject
                    SourceObj = self.SourceObject

                    if self.TotalRefPoints:
                        for RefP in self.TotalRefPoints:
                            bpy.data.objects.remove(RefP)

                    AlignColl = bpy.data.collections.get("ALIGN POINTS")
                    if AlignColl:
                        bpy.data.collections.remove(AlignColl)

                    BDENTAL_4D_Props = context.scene.BDENTAL_4D_Props
                    BDENTAL_4D_Props.AlignModalState = False

                    bpy.ops.screen.screen_full_area(self.FullOverride)

                    if self.ResultMessage:
                        ShowMessageBox(
                            message=self.ResultMessage, icon="COLORSET_03_VEC"
                        )
                    ##########################################################

                    finish = Tcounter()
                    print(f"Alignement finshed in {finish-start} secondes")

                    return {"FINISHED"}

        ###########################################
        elif event.type == ("ESC"):

            if event.value == ("PRESS"):

                ##########################################################
                self.FullSpace3D.overlay.show_outline_selected = True
                self.FullSpace3D.overlay.show_object_origins = True
                self.FullSpace3D.overlay.show_annotation = True
                self.FullSpace3D.overlay.show_text = True
                self.FullSpace3D.overlay.show_extras = True
                self.FullSpace3D.overlay.show_floor = True
                self.FullSpace3D.overlay.show_axis_x = True
                self.FullSpace3D.overlay.show_axis_y = True
                ###########################################################
                for Name in self.visibleObjects:
                    obj = bpy.data.objects.get(Name)
                    if obj:
                        obj.hide_set(False)

                bpy.ops.object.select_all(self.FullOverride, action="DESELECT")
                bpy.ops.wm.tool_set_by_id(self.FullOverride, name="builtin.select")
                bpy.context.scene.tool_settings.use_snap = False
                bpy.context.scene.cursor.location = (0, 0, 0)
                bpy.ops.screen.region_toggle(self.FullOverride, region_type="UI")

                if self.Solid:
                    self.FullSpace3D.shading.background_color = self.background_color
                    self.FullSpace3D.shading.background_type = self.background_type

                TargetObj = self.TargetObject
                SourceObj = self.SourceObject

                if self.TotalRefPoints:
                    for RefP in self.TotalRefPoints:
                        bpy.data.objects.remove(RefP)

                AlignColl = bpy.data.collections.get("ALIGN POINTS")
                if AlignColl:
                    bpy.data.collections.remove(AlignColl)

                BDENTAL_4D_Props = context.scene.BDENTAL_4D_Props
                BDENTAL_4D_Props.AlignModalState = False

                bpy.ops.screen.screen_full_area(self.FullOverride)

                message = [
                    " The Align Operation was Cancelled!",
                ]

                ShowMessageBox(message=message, icon="COLORSET_03_VEC")

                return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        Condition_1 = len(bpy.context.selected_objects) != 2
        Condition_2 = bpy.context.selected_objects and not bpy.context.active_object
        Condition_3 = bpy.context.selected_objects and not (
            bpy.context.active_object in bpy.context.selected_objects
        )

        if Condition_1 or Condition_2 or Condition_3:

            message = [
                "Selection is invalid !",
                "Please Deselect all objects,",
                "Select the Object to Align and ,",
                "<SHIFT + Select> the Base Object.",
                "Click info button for more info.",
            ]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:

            if context.space_data.type == "VIEW_3D":
                BDENTAL_4D_Props = context.scene.BDENTAL_4D_Props
                BDENTAL_4D_Props.AlignModalState = True
                # Prepare scene  :
                ##########################################################

                bpy.context.space_data.overlay.show_outline_selected = False
                bpy.context.space_data.overlay.show_object_origins = False
                bpy.context.space_data.overlay.show_annotation = False
                bpy.context.space_data.overlay.show_text = False
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
                self.TargetObject = bpy.context.active_object
                self.SourceObject = [
                    obj
                    for obj in bpy.context.selected_objects
                    if not obj is self.TargetObject
                ][0]

                VisObj = bpy.context.visible_objects
                self.visibleObjects = [obj.name for obj in VisObj]
                for obj in VisObj:
                    if not obj in [self.TargetObject, self.SourceObject]:
                        obj.hide_set(True)

                self.Solid = False
                if bpy.context.space_data.shading.type == "SOLID":
                    self.Solid = True
                    self.background_type = (
                        bpy.context.space_data.shading.background_type
                    )
                    bpy.context.space_data.shading.background_type = "VIEWPORT"
                    self.background_color = tuple(
                        bpy.context.space_data.shading.background_color
                    )
                    bpy.context.space_data.shading.background_color = (0.0, 0.0, 0.0)

                self.TargetVoxelMode = self.TargetObject.name.startswith(
                    "BD"
                ) and self.TargetObject.name.endswith("_CTVolume")
                self.SourceVoxelMode = self.SourceObject.name.startswith(
                    "BD"
                ) and self.SourceObject.name.endswith("_CTVolume")
                self.TargetRefPoints = []
                self.SourceRefPoints = []
                self.TotalRefPoints = []

                self.TargetCounter = 0
                self.SourceCounter = 0

                bpy.ops.screen.screen_full_area()
                self.FullOverride, self.FullArea3D, self.FullSpace3D = CtxOverride(
                    context
                )

                context.window_manager.modal_handler_add(self)

                return {"RUNNING_MODAL"}

            else:

                self.report({"WARNING"}, "Active space must be a View3d")

                return {"CANCELLED"}


############################################################################
class BDENTAL_4D_OT_AlignPointsInfo(bpy.types.Operator):
    """ Add Align Refference points """

    bl_idname = "bdental4d.alignpointsinfo"
    bl_label = "INFO"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        message = [
            "\u2588 Deselect all objects,",
            "\u2588 Select the Object to Align,",
            "\u2588 Press <SHIFT + Click> to select the Base Object,",
            "\u2588 Click <ALIGN> button,",
            f"      Press <Left Click> to Place Cursor,",
            f"      Press <'B'> to Add Green Point (Base),",
            f"      Press <'A'> to Add Red Point (Align),",
            f"      Press <'DEL'> to delete Point,",
            f"      Press <'ESC'> to Cancel Operation,",
            f"      Press <'ENTER'> to execute Alignement.",
            "\u2588 NOTE :",
            "3 Green Points and 3 Red Points,",
            "are the minimum required for Alignement!",
        ]
        ShowMessageBox(message=message, title="INFO", icon="INFO")

        return {"FINISHED"}


########################################################################
# Mesh Tools Operators
########################################################################
class BDENTAL_4D_OT_AddColor(bpy.types.Operator):
    """Add color material """

    bl_idname = "bdental4d.add_color"
    bl_label = "Add Color"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = context.active_object
        if not obj:

            message = ["Please select target object !"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")
            return {"CANCELLED"}

        if not obj.select_get() or not obj.type in ["MESH", "CURVE"]:
            message = ["Please select target object !"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:

            matName = f"{obj.name}_Mat"
            mat = bpy.data.materials.get(matName) or bpy.data.materials.new(matName)
            mat.use_nodes = False
            mat.diffuse_color = [0.8, 0.8, 0.8, 1.0]

            obj.active_material = mat

        return {"FINISHED"}


class BDENTAL_4D_OT_RemoveColor(bpy.types.Operator):
    """Remove color material """

    bl_idname = "bdental4d.remove_color"
    bl_label = "Remove Color"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = context.active_object
        if not obj:

            message = ["Please select target object !"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")
            return {"CANCELLED"}

        if not obj.select_get() or not obj.type in ["MESH", "CURVE"]:
            message = ["Please select target object !"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:

            if obj.material_slots:
                for _ in obj.material_slots:
                    bpy.ops.object.material_slot_remove()

        return {"FINISHED"}


class BDENTAL_4D_OT_JoinObjects(bpy.types.Operator):
    " Join Objects "

    bl_idname = "bdental4d.join_objects"
    bl_label = "JOIN :"

    def execute(self, context):

        ActiveObj = context.active_object
        condition = (
            ActiveObj
            and ActiveObj in context.selected_objects
            and len(context.selected_objects) >= 2
        )

        if not condition:

            message = [" Please select objects to join !"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:

            bpy.ops.object.join()

            return {"FINISHED"}


#######################################################################################
# Separate models operator :


class BDENTAL_4D_OT_SeparateObjects(bpy.types.Operator):
    " Separate Objects "

    bl_idname = "bdental4d.separate_objects"
    bl_label = "SEPARATE :"

    Separate_Modes_List = ["Selection", "Loose Parts", ""]
    items = []
    for i in range(len(Separate_Modes_List)):
        item = (
            str(Separate_Modes_List[i]),
            str(Separate_Modes_List[i]),
            str(""),
            int(i),
        )
        items.append(item)

    SeparateMode: EnumProperty(
        items=items, description="SeparateMode", default="Loose Parts"
    )

    def execute(self, context):

        if self.SeparateMode == "Loose Parts":
            bpy.ops.mesh.separate(type="LOOSE")

        if self.SeparateMode == "Selection":
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.separate(type="SELECTED")

        bpy.ops.object.mode_set(mode="OBJECT")

        Parts = list(context.selected_objects)

        if Parts and len(Parts) > 1:
            for obj in Parts:
                bpy.ops.object.select_all(action="DESELECT")
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")

            bpy.ops.object.select_all(action="DESELECT")
            Parts[-1].select_set(True)
            bpy.context.view_layer.objects.active = Parts[-1]

        return {"FINISHED"}

    def invoke(self, context, event):

        self.ActiveObj = context.active_object
        condition = (
            self.ActiveObj
            and self.ActiveObj.type == "MESH"
            and self.ActiveObj in bpy.context.selected_objects
        )

        if not condition:

            message = [" Please select the target object !"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:
            wm = context.window_manager
            return wm.invoke_props_dialog(self)


#######################################################################################
# Parent model operator :


class BDENTAL_4D_OT_Parent(bpy.types.Operator):
    " Parent Object "

    bl_idname = "bdental4d.parent_object"
    bl_label = "PARENT"

    def execute(self, context):

        ActiveObj = context.active_object
        condition = (
            ActiveObj
            and ActiveObj in context.selected_objects
            and len(context.selected_objects) >= 2
        )

        if not condition:
            message = [
                " Please select child objects,",
                "parent object should be,",
                "the last one selected!",
            ]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:
            parent = ActiveObj
            bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)
            bpy.ops.object.select_all(action="DESELECT")
            parent.select_set(True)
            # bpy.ops.object.hide_view_set(unselected=True)

            return {"FINISHED"}


#######################################################################################
# Unparent model operator :


class BDENTAL_4D_OT_Unparent(bpy.types.Operator):
    " Un-Parent objects"

    bl_idname = "bdental4d.unparent_objects"
    bl_label = "Un-Parent"

    def execute(self, context):

        ActiveObj = context.active_object
        condition = ActiveObj and ActiveObj in bpy.context.selected_objects

        if not condition:

            message = [" Please select the target objects !"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:
            bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
            bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")

            return {"FINISHED"}


#######################################################################################
# Align model to front operator :


class BDENTAL_4D_OT_align_to_front(bpy.types.Operator):
    """Align Model To Front view"""

    bl_idname = "bdental4d.align_to_front"
    bl_label = "Align to Front"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        ActiveObj = context.active_object
        condition = (
            ActiveObj
            and ActiveObj.type == "MESH"
            and ActiveObj in bpy.context.selected_objects
        )

        if not condition:

            message = [" Please select the target object !"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:

            Model = bpy.context.view_layer.objects.active

            # get object rotation mode and invert it :

            rot_mod = Model.rotation_mode

            # Get VIEW_rotation matrix  :

            view3d_rot_matrix = (
                context.space_data.region_3d.view_rotation.to_matrix().to_4x4()
            )

            # create a 90 degrees arround X_axis Euler :
            Eul_90x = Euler((radians(90), 0, 0), rot_mod)

            # Euler to mattrix 4x4 :
            Eul_90x_matrix = Eul_90x.to_matrix().to_4x4()

            # Rotate Model :
            Model.matrix_world = (
                Eul_90x_matrix @ view3d_rot_matrix.inverted() @ Model.matrix_world
            )
            bpy.ops.view3d.view_all(center=True)
            bpy.ops.view3d.view_axis(type="FRONT")
            bpy.ops.wm.tool_set_by_id(name="builtin.cursor")

        return {"FINISHED"}


#######################################################################################
# Center model modal operator :


class BDENTAL_4D_OT_to_center(bpy.types.Operator):
    " Center Model to world origin "

    bl_idname = "bdental4d.to_center"
    bl_label = "TO CENTER"

    yellow_stone = [1.0, 0.36, 0.06, 1.0]

    def modal(self, context, event):

        if not event.type in {"RET", "ESC"}:
            # allow navigation

            return {"PASS_THROUGH"}

        elif event.type == "RET":

            if event.value == ("PRESS"):

                if context.scene.cursor.location == (0, 0, 0):

                    bpy.ops.object.origin_set(type="ORIGIN_CURSOR", center="MEDIAN")
                    bpy.ops.view3d.snap_selected_to_cursor(use_offset=True)

                else:

                    bpy.ops.object.origin_set(type="ORIGIN_CURSOR", center="MEDIAN")
                    bpy.ops.view3d.snap_cursor_to_center()
                    bpy.ops.view3d.snap_selected_to_cursor(use_offset=True)
                    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")

            bpy.ops.view3d.view_all(center=True)
            # bpy.ops.wm.tool_set_by_id(name="builtin.cursor")
            bpy.ops.wm.tool_set_by_id(name="builtin.select")

            return {"FINISHED"}

        elif event.type == ("ESC"):

            if event.value == ("PRESS"):

                bpy.ops.wm.tool_set_by_id(name="builtin.select")

                return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):

        ActiveObj = context.active_object
        condition = (
            ActiveObj
            and ActiveObj.type == "MESH"
            and ActiveObj in bpy.context.selected_objects
        )

        if not condition:

            message = [" Please select the target object !"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:

            if context.space_data.type == "VIEW_3D":

                bpy.ops.ed.undo_push()

                bpy.ops.object.mode_set(mode="OBJECT")
                Model = bpy.context.view_layer.objects.active
                bpy.ops.object.select_all(action="DESELECT")
                Model.select_set(True)

                bpy.ops.wm.tool_set_by_id(name="builtin.cursor")

                message = [
                    " Please move cursor to incisal Midline",
                    " and click < ENTER >!",
                ]
                ShowMessageBox(message=message, icon="COLORSET_02_VEC")

                context.window_manager.modal_handler_add(self)

                return {"RUNNING_MODAL"}

            else:

                self.report({"WARNING"}, "Active space must be a View3d")

                return {"CANCELLED"}


#######################################################################################
# Cursor to world origin operator :


class BDENTAL_4D_OT_center_cursor(bpy.types.Operator):
    """Cursor to World Origin """

    bl_idname = "bdental4d.center_cursor"
    bl_label = "Center Cursor"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        bpy.ops.view3d.snap_cursor_to_center()

        return {"FINISHED"}


#######################################################################################
# Add Occlusal Plane :
class BDENTAL_4D_OT_OcclusalPlane(bpy.types.Operator):
    """ Add Occlusal Plane"""

    bl_idname = "bdental4d.occlusalplane"
    bl_label = "OCCLUSAL PLANE"
    bl_options = {"REGISTER", "UNDO"}

    CollName = "Occlusal Points"
    OcclusalPoints = []

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
        if event.type == "R":
            # Add Right point :
            if event.value == ("PRESS"):
                Override, area3D, space3D = CtxOverride(context)
                color = (1, 0, 0, 1)  # red
                CollName = self.CollName
                name = "Right_Occlusal_Point"
                OldPoint = bpy.data.objects.get(name)
                if OldPoint:
                    bpy.data.objects.remove(OldPoint)
                loc = context.scene.cursor.location
                NewPoint = AddMarkupPoint(name, color, loc, 1.2, CollName)
                self.RightPoint = NewPoint
                bpy.ops.object.select_all(Override, action="DESELECT")
                self.OcclusalPoints = [
                    name
                    for name in self.OcclusalPoints
                    if not name == "Right_Occlusal_Point"
                ]
                self.OcclusalPoints.append(self.RightPoint.name)

        #########################################
        if event.type == "A":
            # Add Right point :
            if event.value == ("PRESS"):
                Override, area3D, space3D = CtxOverride(context)
                color = (0, 1, 0, 1)  # green
                CollName = self.CollName
                name = "Anterior_Occlusal_Point"
                OldPoint = bpy.data.objects.get(name)
                if OldPoint:
                    bpy.data.objects.remove(OldPoint)
                loc = context.scene.cursor.location
                NewPoint = AddMarkupPoint(name, color, loc, 1.2, CollName)
                self.AnteriorPoint = NewPoint
                bpy.ops.object.select_all(Override, action="DESELECT")

                self.OcclusalPoints = [
                    name
                    for name in self.OcclusalPoints
                    if not name == "Anterior_Occlusal_Point"
                ]
                self.OcclusalPoints.append(self.AnteriorPoint.name)
        #########################################
        if event.type == "L":
            # Add Right point :
            if event.value == ("PRESS"):
                Override, area3D, space3D = CtxOverride(context)
                color = (0, 0, 1, 1)  # blue
                CollName = self.CollName
                name = "Left_Occlusal_Point"
                OldPoint = bpy.data.objects.get(name)
                if OldPoint:
                    bpy.data.objects.remove(OldPoint)
                loc = context.scene.cursor.location
                NewPoint = AddMarkupPoint(name, color, loc, 1.2, CollName)
                self.LeftPoint = NewPoint
                bpy.ops.object.select_all(Override, action="DESELECT")
                self.OcclusalPoints = [
                    name
                    for name in self.OcclusalPoints
                    if not name == "Left_Occlusal_Point"
                ]
                self.OcclusalPoints.append(self.LeftPoint.name)
        #########################################

        elif event.type == ("DEL") and event.value == ("PRESS"):

            if self.OcclusalPoints:
                name = self.OcclusalPoints.pop()
                bpy.data.objects.remove(bpy.data.objects.get(name))

        elif event.type == "RET":
            if event.value == ("PRESS"):

                if not len(self.OcclusalPoints) == 3:
                    message = ["3 points needed", "Please check Info and retry"]
                    ShowMessageBox(message=message, icon="COLORSET_02_VEC")

                else:
                    OcclusalPlane = PointsToOcclusalPlane(
                        self.FullOverride,
                        self.TargetObject,
                        self.RightPoint,
                        self.AnteriorPoint,
                        self.LeftPoint,
                        color=(0.0, 0.0, 0.2, 0.7),
                        subdiv=50,
                    )

                    #########################################################
                    self.FullSpace3D.overlay.show_outline_selected = True
                    self.FullSpace3D.overlay.show_object_origins = True
                    self.FullSpace3D.overlay.show_annotation = True
                    self.FullSpace3D.overlay.show_text = True
                    self.FullSpace3D.overlay.show_extras = True
                    self.FullSpace3D.overlay.show_floor = True
                    self.FullSpace3D.overlay.show_axis_x = True
                    self.FullSpace3D.overlay.show_axis_y = True
                    ##########################################################
                    for Name in self.visibleObjects:
                        obj = bpy.data.objects.get(Name)
                        if obj:
                            obj.hide_set(False)

                    bpy.ops.object.select_all(self.FullOverride, action="DESELECT")
                    bpy.ops.wm.tool_set_by_id(self.FullOverride, name="builtin.select")
                    bpy.context.scene.tool_settings.use_snap = False
                    bpy.context.scene.cursor.location = (0, 0, 0)
                    bpy.ops.screen.region_toggle(self.FullOverride, region_type="UI")

                    self.FullSpace3D.shading.background_color = self.background_color
                    self.FullSpace3D.shading.background_type = self.background_type

                    bpy.ops.screen.screen_full_area(self.FullOverride)

                    if self.OcclusalPoints:
                        for name in self.OcclusalPoints:
                            P = bpy.data.objects.get(name)
                            if P:
                                bpy.data.objects.remove(P)
                    Coll = bpy.data.collections.get(self.CollName)
                    if Coll:
                        bpy.data.collections.remove(Coll)
                    ##########################################################
                    return {"FINISHED"}

        elif event.type == ("ESC"):

            ##########################################################
            self.FullSpace3D.overlay.show_outline_selected = True
            self.FullSpace3D.overlay.show_object_origins = True
            self.FullSpace3D.overlay.show_annotation = True
            self.FullSpace3D.overlay.show_text = True
            self.FullSpace3D.overlay.show_extras = True
            self.FullSpace3D.overlay.show_floor = True
            self.FullSpace3D.overlay.show_axis_x = True
            self.FullSpace3D.overlay.show_axis_y = True
            ###########################################################
            for Name in self.visibleObjects:
                obj = bpy.data.objects.get(Name)
                if obj:
                    obj.hide_set(False)

            bpy.ops.object.select_all(self.FullOverride, action="DESELECT")
            bpy.ops.wm.tool_set_by_id(self.FullOverride, name="builtin.select")
            bpy.context.scene.tool_settings.use_snap = False
            bpy.context.scene.cursor.location = (0, 0, 0)
            bpy.ops.screen.region_toggle(self.FullOverride, region_type="UI")

            self.FullSpace3D.shading.background_color = self.background_color
            self.FullSpace3D.shading.background_type = self.background_type

            bpy.ops.screen.screen_full_area(self.FullOverride)

            if self.OcclusalPoints:
                for name in self.OcclusalPoints:
                    P = bpy.data.objects.get(name)
                    if P:
                        bpy.data.objects.remove(P)
            Coll = bpy.data.collections.get(self.CollName)
            if Coll:
                bpy.data.collections.remove(Coll)

            message = [
                " The Occlusal Plane Operation was Cancelled!",
            ]

            ShowMessageBox(message=message, icon="COLORSET_03_VEC")

            return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):
        Condition_1 = bpy.context.selected_objects and bpy.context.active_object

        if not Condition_1:

            message = [
                "Please select Target object",
            ]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:

            if context.space_data.type == "VIEW_3D":

                # Prepare scene  :
                ##########################################################

                bpy.context.space_data.overlay.show_outline_selected = False
                bpy.context.space_data.overlay.show_object_origins = False
                bpy.context.space_data.overlay.show_annotation = False
                bpy.context.space_data.overlay.show_text = False
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
                self.TargetObject = bpy.context.active_object
                VisObj = bpy.context.visible_objects
                self.visibleObjects = [obj.name for obj in VisObj]

                for obj in VisObj:
                    if obj is not self.TargetObject:
                        obj.hide_set(True)
                self.Background = bpy.context.space_data.shading.type
                bpy.context.space_data.shading.type = "SOLID"
                self.background_type = bpy.context.space_data.shading.background_type
                bpy.context.space_data.shading.background_type = "VIEWPORT"
                self.background_color = tuple(
                    bpy.context.space_data.shading.background_color
                )
                bpy.context.space_data.shading.background_color = (0.0, 0.0, 0.0)
                bpy.ops.screen.region_toggle(region_type="UI")
                bpy.ops.object.select_all(action="DESELECT")
                bpy.ops.screen.screen_full_area()
                self.FullOverride, self.FullArea3D, self.FullSpace3D = CtxOverride(
                    context
                )

                context.window_manager.modal_handler_add(self)

                return {"RUNNING_MODAL"}

            else:

                self.report({"WARNING"}, "Active space must be a View3d")

                return {"CANCELLED"}


############################################################################
class BDENTAL_4D_OT_OcclusalPlaneInfo(bpy.types.Operator):
    """ Add Align Refference points """

    bl_idname = "bdental4d.occlusalplaneinfo"
    bl_label = "INFO"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        message = [
            "\u2588 Deselect all objects,",
            "\u2588 Select the target Object,",
            "\u2588 Click < OCCLUSAL PLANE > button,",
            f"      Press <Left Click> to Place Cursor,",
            f"      <'R'> to Add Right Point,",
            f"      <'A'> to Add Anterior median Point,",
            f"      <'L'> to Add Left Point,",
            f"      Press <'DEL'> to delete Point,",
            f"      Press <'ESC'> to Cancel Operation,",
            f"      Press <'ENTER'> to Add Occlusal Plane.",
        ]
        ShowMessageBox(message=message, title="INFO", icon="INFO")

        return {"FINISHED"}


#######################################################################################
# Decimate model operator :


class BDENTAL_4D_OT_decimate(bpy.types.Operator):
    """ Decimate to ratio """

    bl_idname = "bdental4d.decimate"
    bl_label = "Decimate Model"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        BDENTAL_4D_Props = context.scene.BDENTAL_4D_Props
        decimate_ratio = round(BDENTAL_4D_Props.decimate_ratio, 2)

        ActiveObj = context.active_object
        condition = (
            ActiveObj
            and ActiveObj.type == "MESH"
            and ActiveObj in bpy.context.selected_objects
        )

        if not condition:

            message = [" Please select the target object !"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}
        else:

            bpy.ops.object.mode_set(mode="OBJECT")
            bpy.ops.object.modifier_add(type="DECIMATE")
            bpy.context.object.modifiers["Decimate"].ratio = decimate_ratio
            bpy.ops.object.modifier_apply(modifier="Decimate")

            return {"FINISHED"}


#######################################################################################
# Fill holes operator :


class BDENTAL_4D_OT_fill(bpy.types.Operator):
    """fill edge or face """

    bl_idname = "bdental4d.fill"
    bl_label = "FILL"
    bl_options = {"REGISTER", "UNDO"}

    Fill_treshold: IntProperty(
        name="Hole Fill Treshold",
        description="Hole Fill Treshold",
        default=400,
    )

    def execute(self, context):

        Mode = self.ActiveObj.mode

        if not Mode == "EDIT":
            bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.mesh.edge_face_add()
        bpy.ops.mesh.subdivide(number_cuts=10)
        bpy.ops.mesh.subdivide(number_cuts=2)
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_non_manifold()
        bpy.ops.mesh.remove_doubles(threshold=0.1)

        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_non_manifold()
        bpy.ops.mesh.fill_holes(sides=self.Fill_treshold)
        bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
        bpy.ops.mesh.select_all(action="DESELECT")

        bpy.ops.object.mode_set(mode=Mode)

        return {"FINISHED"}

    def invoke(self, context, event):
        self.ActiveObj = context.active_object
        condition = (
            self.ActiveObj
            and self.ActiveObj.type == "MESH"
            and self.ActiveObj in bpy.context.selected_objects
        )

        if not condition:

            message = [" Please select the target object !"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:
            wm = context.window_manager
            return wm.invoke_props_dialog(self)


#######################################################################################
# Retopo smooth operator :


class BDENTAL_4D_OT_retopo_smooth(bpy.types.Operator):
    """Retopo sculpt for filled holes"""

    bl_idname = "bdental4d.retopo_smooth"
    bl_label = "Retopo Smooth"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        ActiveObj = context.active_object
        condition = (
            ActiveObj
            and ActiveObj.type == "MESH"
            and ActiveObj in bpy.context.selected_objects
        )

        if not condition:

            message = [" Please select the target object !"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:

            # Prepare scene settings :
            bpy.context.tool_settings.mesh_select_mode = (True, False, False)

            bpy.ops.object.mode_set(mode="SCULPT")

            Model = bpy.context.view_layer.objects.active

            bpy.context.scene.tool_settings.sculpt.use_symmetry_x = False
            bpy.context.scene.tool_settings.unified_paint_settings.size = 50

            bpy.ops.wm.tool_set_by_id(name="builtin_brush.Simplify")
            bpy.data.brushes["Simplify"].cursor_color_add = (0.3, 0.0, 0.7, 0.4)
            bpy.data.brushes["Simplify"].strength = 0.5
            bpy.data.brushes["Simplify"].auto_smooth_factor = 0.5
            bpy.data.brushes["Simplify"].use_automasking_topology = True
            bpy.data.brushes["Simplify"].use_frontface = True

            if Model.use_dynamic_topology_sculpting == False:
                bpy.ops.sculpt.dynamic_topology_toggle()

            bpy.context.scene.tool_settings.sculpt.detail_type_method = "CONSTANT"
            bpy.context.scene.tool_settings.sculpt.constant_detail_resolution = 16
            bpy.ops.sculpt.sample_detail_size(mode="DYNTOPO")

            return {"FINISHED"}


#######################################################################################
# clean model operator :
class BDENTAL_4D_OT_clean_mesh2(bpy.types.Operator):
    """ Fill small and medium holes and remove small parts"""

    bl_idname = "bdental4d.clean_mesh2"
    bl_label = "CLEAN MESH"
    bl_options = {"REGISTER", "UNDO"}

    Fill_treshold: IntProperty(
        name="Holes Fill Treshold",
        description="Hole Fill Treshold",
        default=100,
    )

    def execute(self, context):

        ActiveObj = self.ActiveObj

        ####### Get model to clean #######
        bpy.ops.object.mode_set(mode="OBJECT")
        # bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")
        Obj = ActiveObj
        bpy.ops.object.select_all(action="DESELECT")
        Obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)

        ####### Remove doubles, Make mesh consistent (face normals) #######
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=0.1)
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.normals_make_consistent(inside=False)

        ############ clean non_manifold borders ##############
        # bpy.ops.mesh.select_all(action="DESELECT")
        # bpy.ops.mesh.select_non_manifold()
        # bpy.ops.mesh.select_less()
        # bpy.ops.mesh.delete(type="VERT")
        # bpy.ops.mesh.select_all(action="SELECT")
        # bpy.ops.mesh.fill_holes(sides=self.Fill_treshold)
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_non_manifold()
        bpy.ops.mesh.edge_split(type="EDGE")
        bpy.ops.object.mode_set(mode="OBJECT")

        To_Remove = [
            obj for obj in context.selected_objects if not obj.name == Obj.name
        ]
        if To_Remove:
            for obj in To_Remove:
                bpy.data.objects.remove(obj)
        Obj.select_set(True)
        bpy.context.view_layer.objects.active = Obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_non_manifold()
        bpy.ops.mesh.select_less()
        bpy.ops.mesh.delete(type="VERT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.fill_holes(sides=self.Fill_treshold)
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_non_manifold()

        bpy.ops.mesh.looptools_relax(
            input="selected",
            interpolation="cubic",
            iterations="3",
            regular=True,
        )

        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")
        Obj.select_set(True)
        bpy.context.view_layer.objects.active = Obj

        print("Clean Mesh finished.")

        return {"FINISHED"}

    def invoke(self, context, event):
        self.ActiveObj = context.active_object
        condition = (
            self.ActiveObj
            and self.ActiveObj.type == "MESH"
            and self.ActiveObj in bpy.context.selected_objects
        )

        if not condition:

            message = [" Please select the target object !"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:
            wm = context.window_manager
            return wm.invoke_props_dialog(self)


# clean model operator :
class BDENTAL_4D_OT_clean_mesh(bpy.types.Operator):
    """ Fill small and medium holes and remove small parts"""

    bl_idname = "bdental4d.clean_mesh"
    bl_label = "CLEAN MESH"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        ActiveObj = context.active_object

        if not ActiveObj:
            message = [" Invalid Selection ", "Please select Target mesh ! "]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")
            return {"CANCELLED"}
        else:
            Conditions = [
                not ActiveObj.select_set,
                not ActiveObj.type == "MESH",
            ]

            if Conditions[0] or Conditions[1]:
                message = [" Invalid Selection ", "Please select Target mesh ! "]
                ShowMessageBox(message=message, icon="COLORSET_02_VEC")
                return {"CANCELLED"}

            else:

                ####### Get model to clean #######
                bpy.ops.object.mode_set(mode="OBJECT")
                # bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")
                Obj = ActiveObj
                bpy.ops.object.select_all(action="DESELECT")
                Obj.select_set(True)
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.context.tool_settings.mesh_select_mode = (True, False, False)

                ####### Remove doubles, Make mesh consistent (face normals) #######
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.remove_doubles(threshold=0.1)
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.normals_make_consistent(inside=False)

                ############ clean non_manifold borders ##############
                bpy.ops.mesh.select_all(action="DESELECT")
                bpy.ops.mesh.select_non_manifold()
                bpy.ops.mesh.delete(type="VERT")

                bpy.ops.mesh.select_non_manifold()
                bpy.ops.mesh.select_more()
                bpy.ops.mesh.select_less()
                bpy.ops.mesh.delete(type="VERT")

                ####### Fill Holes #######

                bpy.ops.mesh.select_all(action="DESELECT")
                bpy.ops.mesh.select_non_manifold()
                bpy.ops.mesh.fill_holes(sides=100)
                bpy.ops.mesh.quads_convert_to_tris(
                    quad_method="BEAUTY", ngon_method="BEAUTY"
                )

                ####### Relax borders #######
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_all(action="DESELECT")
                bpy.ops.mesh.select_non_manifold()
                bpy.ops.mesh.remove_doubles(threshold=0.1)

                bpy.ops.mesh.looptools_relax(
                    input="selected",
                    interpolation="cubic",
                    iterations="1",
                    regular=True,
                )

                bpy.ops.mesh.select_all(action="DESELECT")
                bpy.ops.object.mode_set(mode="OBJECT")
                Obj.select_set(True)
                bpy.context.view_layer.objects.active = Obj

                print("Clean Mesh finished.")

                return {"FINISHED"}


class BDENTAL_4D_OT_VoxelRemesh(bpy.types.Operator):
    """ Voxel Remesh Operator """

    bl_idname = "bdental4d.voxelremesh"
    bl_label = "REMESH"
    bl_options = {"REGISTER", "UNDO"}

    VoxelSize: FloatProperty(
        name="Voxel Size",
        description="Remesh Voxel Size",
        default=0.1,
        min=0.0,
        max=100.0,
        soft_min=0.0,
        soft_max=100.0,
        step=10,
        precision=1,
    )

    def execute(self, context):
        ActiveObj = context.active_object
        # get model to clean :
        bpy.ops.object.mode_set(mode="OBJECT")
        ActiveObj.data.remesh_mode = "VOXEL"
        ActiveObj.data.remesh_voxel_size = self.VoxelSize
        ActiveObj.data.use_remesh_fix_poles = True
        ActiveObj.data.use_remesh_smooth_normals = True
        ActiveObj.data.use_remesh_preserve_volume = True

        bpy.ops.object.voxel_remesh()
        return {"FINISHED"}

    def invoke(self, context, event):

        ActiveObj = context.active_object
        if not ActiveObj:

            message = [" Please select the target object !"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:

            condition = ActiveObj.type == "MESH" and ActiveObj.select_get() == True

            if not condition:

                message = [" Please select the target object !"]
                ShowMessageBox(message=message, icon="COLORSET_02_VEC")

                return {"CANCELLED"}

            else:
                self.ActiveObj = ActiveObj
                self.VoxelSize = 0.1
                wm = context.window_manager
                return wm.invoke_props_dialog(self)


class BDENTAL_4D_OT_SplintCutterAdd(bpy.types.Operator):
    """  """

    bl_idname = "bdental4d.splintcutteradd"
    bl_label = "CURVE CUTTER ADD"
    bl_options = {"REGISTER", "UNDO"}

    def modal(self, context, event):

        BDENTAL_4D_Props = context.scene.BDENTAL_4D_Props

        if not event.type in {
            "DEL",
            "LEFTMOUSE",
            "RET",
            "ESC",
        }:
            # allow navigation

            return {"PASS_THROUGH"}

        elif event.type == ("DEL"):
            if event.value == ("PRESS"):

                DeleteLastCurvePoint()

            return {"RUNNING_MODAL"}

        elif event.type == ("LEFTMOUSE"):

            if event.value == ("PRESS"):

                return {"PASS_THROUGH"}

            if event.value == ("RELEASE"):

                ExtrudeCurvePointToCursor(context, event)

        elif event.type == "RET":

            if event.value == ("PRESS"):
                CurveCutterName = BDENTAL_4D_Props.CurveCutterNameProp
                CurveCutter = bpy.data.objects[CurveCutterName]
                CurveCutter.select_set(True)
                bpy.context.view_layer.objects.active = CurveCutter
                bpy.ops.object.mode_set(mode="OBJECT")

                if BDENTAL_4D_Props.CurveCutCloseMode == "Close Curve":
                    bpy.ops.object.mode_set(mode="EDIT")
                    bpy.ops.curve.cyclic_toggle()
                    bpy.ops.object.mode_set(mode="OBJECT")

                bpy.ops.wm.tool_set_by_id(name="builtin.select")
                bpy.context.scene.tool_settings.use_snap = False
                bpy.context.space_data.overlay.show_outline_selected = True

                return {"FINISHED"}

        elif event.type == ("ESC"):

            if event.value == ("PRESS"):

                CurveCutterName = bpy.context.scene.BDENTAL_4D_Props.CurveCutterNameProp
                CurveCutter = bpy.data.objects[CurveCutterName]
                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.data.objects.remove(CurveCutter)

                CuttingTargetName = context.scene.BDENTAL_4D_Props.CuttingTargetNameProp
                CuttingTarget = bpy.data.objects[CuttingTargetName]

                bpy.ops.object.select_all(action="DESELECT")
                CuttingTarget.select_set(True)
                bpy.context.view_layer.objects.active = CuttingTarget

                bpy.ops.wm.tool_set_by_id(name="builtin.select")
                bpy.context.scene.tool_settings.use_snap = False
                bpy.context.space_data.overlay.show_outline_selected = True

                return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):

        ActiveObj = context.active_object
        if not ActiveObj:

            message = [" Please select the target object !"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:

            condition = ActiveObj.type == "MESH" and ActiveObj.select_get() == True

            if not condition:

                message = [" Please select the target object !"]
                ShowMessageBox(message=message, icon="COLORSET_02_VEC")

                return {"CANCELLED"}

            else:

                if context.space_data.type == "VIEW_3D":
                    BDENTAL_4D_Props = context.scene.BDENTAL_4D_Props
                    # Assign Model name to CuttingTarget property :
                    CuttingTarget = context.active_object
                    BDENTAL_4D_Props.CuttingTargetNameProp = CuttingTarget.name

                    bpy.ops.object.mode_set(mode="OBJECT")
                    bpy.ops.object.hide_view_set(unselected=True)

                    CuttingCurveAdd()
                    SplintCutter = context.object
                    SplintCutter.name = "BDENTAL_4D_Splint_Cut"
                    BDENTAL_4D_Props.CurveCutterNameProp = "BDENTAL_4D_Splint_Cut"
                    SplintCutter.data.bevel_depth = 0.3
                    SplintCutter.data.extrude = 4
                    SplintCutter.data.offset = -0.3
                    SplintCutter.active_material.diffuse_color = [1, 0, 0, 1]

                    context.window_manager.modal_handler_add(self)

                    return {"RUNNING_MODAL"}

                else:

                    self.report({"WARNING"}, "Active space must be a View3d")

                    return {"CANCELLED"}


class BDENTAL_4D_OT_SplintCutterCut(bpy.types.Operator):
    """  """

    bl_idname = "bdental4d.splintcuttercut"
    bl_label = "CURVE CUTTER ADD"
    bl_options = {"REGISTER", "UNDO"}

    Cut_Modes_List = ["Remove Small Part", "Remove Big Part", "Keep All"]
    items = []
    for i in range(len(Cut_Modes_List)):
        item = (str(Cut_Modes_List[i]), str(Cut_Modes_List[i]), str(""), int(i))
        items.append(item)

    CutMode: EnumProperty(
        name="Splint Cut Mode",
        items=items,
        description="Splint Cut Mode",
        default="Keep All",
    )

    def execute(self, context):

        # Get CurveCutter :
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")

        CurveMeshesList = []
        for CurveCutter in self.CurveCuttersList:
            bpy.ops.object.select_all(action="DESELECT")
            CurveCutter.select_set(True)
            bpy.context.view_layer.objects.active = CurveCutter

            # remove material :
            for mat_slot in CurveCutter.material_slots:
                bpy.ops.object.material_slot_remove()

            # convert CurveCutter to mesh :
            bpy.ops.object.convert(target="MESH")
            CurveMesh = context.object
            CurveMeshesList.append(CurveMesh)

        bpy.ops.object.select_all(action="DESELECT")
        for obj in CurveMeshesList:
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj

        bpy.ops.object.join()
        CurveCutter = context.object
        bpy.ops.object.voxel_remesh()

        bpy.ops.object.select_all(action="DESELECT")
        self.CuttingTarget.select_set(True)
        bpy.context.view_layer.objects.active = self.CuttingTarget

        bpy.ops.object.modifier_add(type="BOOLEAN")
        bpy.context.object.modifiers["Boolean"].show_viewport = False
        bpy.context.object.modifiers["Boolean"].operation = "DIFFERENCE"
        bpy.context.object.modifiers["Boolean"].object = CurveCutter
        bpy.context.object.modifiers["Boolean"].solver = "FAST"
        bpy.ops.object.modifier_apply(modifier="Boolean")

        bpy.data.objects.remove(CurveCutter)

        VisObj = [obj.name for obj in context.visible_objects]
        bpy.ops.object.select_all(action="DESELECT")
        self.CuttingTarget.select_set(True)
        bpy.ops.object.hide_view_set(unselected=True)

        bpy.ops.bdental4d.separate_objects(SeparateMode="Loose Parts")

        if not self.CutMode == "Keep All":

            Splint_Max = (
                max(
                    [
                        [len(obj.data.polygons), obj.name]
                        for obj in context.visible_objects
                    ]
                )
            )[1]
            Splint_min = (
                min(
                    [
                        [len(obj.data.polygons), obj.name]
                        for obj in context.visible_objects
                    ]
                )
            )[1]

            if self.CutMode == "Remove Small Part":
                Splint = bpy.data.objects.get(Splint_Max)
                for obj in context.visible_objects:
                    if not obj is Splint:
                        bpy.data.objects.remove(obj)

            if self.CutMode == "Remove Big Part":
                Splint = bpy.data.objects.get(Splint_min)
                for obj in context.visible_objects:
                    if not obj is Splint:
                        bpy.data.objects.remove(obj)

            Splint.select_set(True)
            bpy.context.view_layer.objects.active = Splint
            bpy.ops.object.shade_flat()

        if self.CutMode == "Keep All":
            bpy.ops.object.select_all(action="DESELECT")

        for objname in VisObj:
            obj = bpy.data.objects.get(objname)
            if obj:
                obj.hide_set(False)

        bpy.context.scene.tool_settings.use_snap = False
        bpy.ops.view3d.snap_cursor_to_center()
        self.BDENTAL_4D_Props.CuttingTargetNameProp = ""
        return {"FINISHED"}

    def invoke(self, context, event):

        self.BDENTAL_4D_Props = context.scene.BDENTAL_4D_Props

        # Get CuttingTarget :
        CuttingTargetName = self.BDENTAL_4D_Props.CuttingTargetNameProp
        self.CuttingTarget = bpy.data.objects.get(CuttingTargetName)
        self.CurveCuttersList = [
            obj
            for obj in context.scene.objects
            if obj.type == "CURVE" and obj.name.startswith("BDENTAL_4D_Splint_Cut")
        ]

        if not self.CurveCuttersList or not self.CuttingTarget:

            message = [" Please Add Splint Cutters first !"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:
            wm = context.window_manager
            return wm.invoke_props_dialog(self)


#######################################################################################
###################################### Cutters ########################################
#######################################################################################
# CurveCutter_01
class BDENTAL_4D_OT_CurveCutterAdd(bpy.types.Operator):
    """ description of this Operator """

    bl_idname = "bdental4d.curvecutteradd"
    bl_label = "CURVE CUTTER ADD"
    bl_options = {"REGISTER", "UNDO"}

    def modal(self, context, event):

        BDENTAL_4D_Props = context.scene.BDENTAL_4D_Props

        if not event.type in {
            "DEL",
            "LEFTMOUSE",
            "RET",
            "ESC",
        }:
            # allow navigation

            return {"PASS_THROUGH"}

        elif event.type == ("DEL"):
            if event.value == ("PRESS"):

                DeleteLastCurvePoint()

            return {"RUNNING_MODAL"}

        elif event.type == ("LEFTMOUSE"):

            if event.value == ("PRESS"):

                return {"PASS_THROUGH"}

            if event.value == ("RELEASE"):

                ExtrudeCurvePointToCursor(context, event)

        elif event.type == "RET":

            if event.value == ("PRESS"):
                CurveCutterName = BDENTAL_4D_Props.CurveCutterNameProp
                CurveCutter = bpy.data.objects[CurveCutterName]
                CurveCutter.select_set(True)
                bpy.context.view_layer.objects.active = CurveCutter
                bpy.ops.object.mode_set(mode="OBJECT")

                if BDENTAL_4D_Props.CurveCutCloseMode == "Close Curve":
                    bpy.ops.object.mode_set(mode="EDIT")
                    bpy.ops.curve.cyclic_toggle()
                    bpy.ops.object.mode_set(mode="OBJECT")

                # bpy.ops.object.modifier_apply(apply_as="DATA", modifier="Shrinkwrap")

                bpy.context.object.data.bevel_depth = 0
                bpy.context.object.data.extrude = 2
                bpy.context.object.data.offset = 0

                bpy.ops.wm.tool_set_by_id(name="builtin.select")
                bpy.context.scene.tool_settings.use_snap = False
                bpy.context.space_data.overlay.show_outline_selected = True

                return {"FINISHED"}

        elif event.type == ("ESC"):

            if event.value == ("PRESS"):

                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.data.objects.remove(self.Cutter)
                Coll = bpy.data.collections.get("BDENTAL-4D Cutters")
                if Coll:
                    Hooks = [obj for obj in Coll.objects if "Hook" in obj.name]
                    if Hooks:
                        for obj in Hooks:
                            bpy.data.objects.remove(obj)
                CuttingTargetName = context.scene.BDENTAL_4D_Props.CuttingTargetNameProp
                CuttingTarget = bpy.data.objects[CuttingTargetName]

                bpy.ops.object.select_all(action="DESELECT")
                CuttingTarget.select_set(True)
                bpy.context.view_layer.objects.active = CuttingTarget

                bpy.ops.wm.tool_set_by_id(name="builtin.select")
                bpy.context.scene.tool_settings.use_snap = False
                bpy.context.space_data.overlay.show_outline_selected = True
                context.scene.BDENTAL_4D_Props.CuttingTargetNameProp = ""
                return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):

        if bpy.context.selected_objects == []:

            message = [" Please select the target object !"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:

            if context.space_data.type == "VIEW_3D":

                # Assign Model name to CuttingTarget property :
                CuttingTarget = bpy.context.view_layer.objects.active
                bpy.context.scene.BDENTAL_4D_Props.CuttingTargetNameProp = (
                    CuttingTarget.name
                )

                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.select_all(action="DESELECT")
                CuttingTarget.select_set(True)
                bpy.context.view_layer.objects.active = CuttingTarget
                # Hide everything but model :
                bpy.ops.object.hide_view_set(unselected=True)

                CuttingCurveAdd()
                self.Cutter = context.active_object

                context.window_manager.modal_handler_add(self)

                return {"RUNNING_MODAL"}

            else:

                self.report({"WARNING"}, "Active space must be a View3d")

                return {"CANCELLED"}


class BDENTAL_4D_OT_CurveCutterCut(bpy.types.Operator):
    " Performe Curve Cutting Operation"

    bl_idname = "bdental4d.curvecuttercut"
    bl_label = "CURVE CUTTER CUT"

    def execute(self, context):

        # Get CuttingTarget :
        CuttingTargetName = bpy.context.scene.BDENTAL_4D_Props.CuttingTargetNameProp
        CuttingTarget = bpy.data.objects[CuttingTargetName]

        # Get CurveCutter :
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")

        CurveCuttersList = [
            obj
            for obj in context.visible_objects
            if obj.type == "CURVE" and obj.name.startswith("BDENTAL4D_Curve_Cut") 
        ]

        if not CurveCuttersList:

            message = [
                " Can't find curve Cutters ",
                "Please ensure curve Cutters are not hiden !",
            ]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        if CurveCuttersList:
            CurveMeshesList = []
            for CurveCutter in CurveCuttersList:
                bpy.ops.object.select_all(action="DESELECT")
                CurveCutter.select_set(True)
                bpy.context.view_layer.objects.active = CurveCutter

                # remove material :
                for mat_slot in CurveCutter.material_slots:
                    bpy.ops.object.material_slot_remove()

                # # Change CurveCutter setting   :
                # bpy.context.object.data.bevel_depth = 0
                # bpy.context.object.data.offset = 0

                # subdivide curve points :

                # bpy.ops.object.mode_set(mode="EDIT")
                # bpy.ops.curve.select_all(action="SELECT")
                # bpy.ops.curve.subdivide()

                # convert CurveCutter to mesh :
                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.convert(target="MESH")
                CurveMesh = context.object
                CurveMeshesList.append(CurveMesh)

            bpy.ops.object.select_all(action="DESELECT")
            for obj in CurveMeshesList:
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
        if len(CurveMeshesList) > 1:
            bpy.ops.object.join()

        CurveCutter = context.object

        CurveCutter.select_set(True)
        bpy.context.view_layer.objects.active = CurveCutter

        bpy.context.scene.tool_settings.use_snap = False
        bpy.ops.view3d.snap_cursor_to_center()

        # # Make vertex group :
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.mesh.select_all(action="SELECT")
        curve_vgroup = CurveCutter.vertex_groups.new(name="curve_vgroup")
        bpy.ops.object.vertex_group_assign()
        bpy.ops.object.mode_set(mode="OBJECT")

        # select CuttingTarget :
        bpy.ops.object.select_all(action="DESELECT")
        CuttingTarget.select_set(True)
        bpy.context.view_layer.objects.active = CuttingTarget

        # delete old vertex groups :
        CuttingTarget.vertex_groups.clear()

        # deselect all vertices :
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="VERT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")

        # Join CurveCutter to CuttingTarget :
        CurveCutter.select_set(True)
        bpy.ops.object.join()
        bpy.ops.object.hide_view_set(unselected=True)

        # intersect make vertex group :
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="VERT")
        bpy.ops.mesh.intersect()

        intersect_vgroup = CuttingTarget.vertex_groups.new(name="intersect_vgroup")
        CuttingTarget.vertex_groups.active_index = intersect_vgroup.index
        bpy.ops.object.vertex_group_assign()

        # OtherObjList = [obj for obj in bpy.data.objects if obj!= CuttingTarget]
        # hide all but object
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.hide_view_set(unselected=True)

        # delete curve_vgroup :
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")
        curve_vgroup = CuttingTarget.vertex_groups["curve_vgroup"]

        CuttingTarget.vertex_groups.active_index = curve_vgroup.index
        bpy.ops.object.vertex_group_select()
        bpy.ops.mesh.delete(type="FACE")

        bpy.ops.ed.undo_push()
        # 1st methode :
        SplitSeparator(CuttingTarget=CuttingTarget)

        for obj in context.visible_objects:
            if len(obj.data.polygons) <= 1:
                bpy.data.objects.remove(obj)

        print("Cutting done with first method")

        # Filtring loose parts :
        # resulting_parts = PartsFilter()

        # if resulting_parts > 1:
        #     for obj in bpy.context.visible_objects:
        #         obj.vertex_groups.clear()

        #     print("Cutting done with first method")

        # else:
        #     pass
        # # bpy.ops.ed.undo()
        # # Get CuttingTarget :
        # CuttingTargetName = bpy.context.scene.BDENTAL_4D_Props.CuttingTargetNameProp
        # CuttingTarget = bpy.data.objects[CuttingTargetName]
        # CuttingTarget.select_set(True)
        # bpy.context.view_layer.objects.active = CuttingTarget

        # bol = True

        # while bol:
        #     bol = IterateSeparator()

        # # Filtring loose parts :
        # resulting_parts = PartsFilter()
        # print("Cutting done with second method")

        # bpy.ops.object.select_all(action="DESELECT")
        # ob = bpy.context.visible_objects[-1]
        # ob.select_set(True)
        # bpy.context.view_layer.objects.active = ob
        # bpy.ops.wm.tool_set_by_id(name="builtin.select")

        return {"FINISHED"}


#######################################################################################

##################################################################


class BDENTAL_4D_OT_AddTube(bpy.types.Operator):
    """ Add Curve Tube """

    bl_idname = "bdental4d.add_tube"
    bl_label = "ADD TUBE"
    bl_options = {"REGISTER", "UNDO"}

    def modal(self, context, event):

        if not event.type in {
            "DEL",
            "LEFTMOUSE",
            "RET",
            "ESC",
        }:
            # allow navigation

            return {"PASS_THROUGH"}

        elif event.type == ("DEL"):
            if event.value == ("PRESS"):

                DeleteTubePoint(self.TubeObject)

            return {"RUNNING_MODAL"}

        elif event.type == ("LEFTMOUSE"):

            if event.value == ("PRESS"):

                return {"PASS_THROUGH"}

            if event.value == ("RELEASE"):

                ExtrudeTube(self.TubeObject)

        elif event.type == "RET":

            if event.value == ("PRESS"):
                bpy.ops.object.mode_set(mode="OBJECT")
                self.TubeObject.select_set(True)
                bpy.context.view_layer.objects.active = self.TubeObject

                if self.TubeCloseMode:
                    bpy.ops.object.mode_set(mode="EDIT")
                    bpy.ops.curve.cyclic_toggle()
                    bpy.ops.object.mode_set(mode="OBJECT")

                bpy.ops.wm.tool_set_by_id(name="builtin.select")
                bpy.context.scene.tool_settings.use_snap = False
                bpy.context.space_data.overlay.show_outline_selected = True

                return {"FINISHED"}

        elif event.type == ("ESC"):

            if event.value == ("PRESS"):

                bpy.ops.object.mode_set(mode="OBJECT")
                Tube = bpy.data.objects.get(self.TubeName)
                if Tube:
                    bpy.data.objects.remove(Tube)

                bpy.ops.object.select_all(action="DESELECT")
                self.TubeTarget.select_set(True)
                bpy.context.view_layer.objects.active = self.TubeTarget

                bpy.ops.wm.tool_set_by_id(name="builtin.select")
                bpy.context.scene.tool_settings.use_snap = False
                bpy.context.space_data.overlay.show_outline_selected = True

                return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):

        self.TubeTarget = context.active_object
        if not self.TubeTarget:
            message = [" Please select the target object !"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:
            if not self.TubeTarget.select_get():

                message = [" Please select the target object !"]
                ShowMessageBox(message=message, icon="COLORSET_02_VEC")

                return {"CANCELLED"}

            else:

                if context.space_data.type == "VIEW_3D":

                    BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props

                    bpy.ops.object.mode_set(mode="OBJECT")
                    bpy.ops.object.select_all(action="DESELECT")

                    self.TubeObject = AddTube(context, self.TubeTarget)
                    self.TubeName = self.TubeObject.name

                    if BDENTAL_4D_Props.TubeCloseMode == "Close Tube":
                        self.TubeCloseMode = True

                    if BDENTAL_4D_Props.TubeCloseMode == "Open Tube":
                        self.TubeCloseMode = False
                        self.TubeObject.data.use_fill_caps = True

                    context.window_manager.modal_handler_add(self)

                    return {"RUNNING_MODAL"}

                else:

                    message = ["Active space must be a View3d"]
                    ShowMessageBox(message=message, icon="COLORSET_02_VEC")

                    return {"CANCELLED"}


#########################################################################################
# CurveCutter_02
class BDENTAL_4D_OT_CurveCutterAdd2(bpy.types.Operator):
    """ description of this Operator """

    bl_idname = "bdental4d.curvecutteradd2"
    bl_label = "CURVE CUTTER ADD"
    bl_options = {"REGISTER", "UNDO"}

    def modal(self, context, event):

        BDENTAL_4D_Props = context.scene.BDENTAL_4D_Props

        if not event.type in {
            "DEL",
            "LEFTMOUSE",
            "RET",
            "ESC",
        }:
            # allow navigation

            return {"PASS_THROUGH"}

        elif event.type == ("DEL"):
            if event.value == ("PRESS"):

                DeleteLastCurvePoint()

            return {"RUNNING_MODAL"}

        elif event.type == ("LEFTMOUSE"):

            if event.value == ("PRESS"):

                return {"PASS_THROUGH"}

            if event.value == ("RELEASE"):

                ExtrudeCurvePointToCursor(context, event)

        elif event.type == "RET":

            if event.value == ("PRESS"):
                
                CuttingTargetName = BDENTAL_4D_Props.CuttingTargetNameProp
                CuttingTarget = bpy.data.objects[CuttingTargetName]

                CurveCutterName = BDENTAL_4D_Props.CurveCutterNameProp
                CurveCutter = bpy.data.objects[CurveCutterName]

                CurveCutter.select_set(True)
                bpy.context.view_layer.objects.active = CurveCutter

                bpy.ops.object.mode_set(mode="OBJECT")

                if BDENTAL_4D_Props.CurveCutCloseMode == "Close Curve":
                    bpy.ops.object.mode_set(mode="EDIT")
                    bpy.ops.curve.cyclic_toggle()
                    bpy.ops.object.mode_set(mode="OBJECT")

                # bpy.ops.object.modifier_apply(apply_as="DATA", modifier="Shrinkwrap")

                bpy.ops.wm.tool_set_by_id(name="builtin.select")
                # bpy.context.scene.tool_settings.use_snap = False
                bpy.context.space_data.overlay.show_outline_selected = True

                bezier_points = CurveCutter.data.splines[0].bezier_points[:]
                Hooks = [obj for obj in bpy.data.objects if 'Hook' in obj.name]
                for i in range(len(bezier_points)):
                    Hook = AddCurveSphere(
                        Name=f"Hook_{i}",
                        Curve=CurveCutter,
                        i=i,
                        CollName="BDENTAL-4D Cutters",
                    )
                    Hooks.append(Hook)
                print(Hooks)
                for h in Hooks :
                    for o in Hooks :
                        if not o is h :
                            delta = o.location - h.location
                            distance = sqrt(delta[0]**2+delta[1]**2+delta[2]**2)
                            if distance<=0.5 :
                                o.location = h.location 
                bpy.context.space_data.overlay.show_relationship_lines = False
                bpy.context.scene.tool_settings.use_snap = True
                bpy.context.scene.tool_settings.snap_elements = {'FACE'}
                bpy.context.scene.tool_settings.snap_target = 'CENTER'
                bpy.ops.object.select_all(action="DESELECT")

                CurveCutter.hide_select = True


                return {"FINISHED"}

        elif event.type == ("ESC"):

            if event.value == ("PRESS"):

                bpy.data.objects.remove(self.CurveCutter)
                bpy.ops.object.select_all(action="DESELECT")
                self.CuttingTarget.select_set(True)
                bpy.context.view_layer.objects.active = self.CuttingTarget

                bpy.ops.wm.tool_set_by_id(name="builtin.select")
                bpy.context.scene.tool_settings.use_snap = False
                bpy.context.space_data.overlay.show_outline_selected = True
                bpy.context.space_data.overlay.show_relationship_lines = False

                return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):

        if bpy.context.selected_objects == []:

            message = [" Please select the target object !"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:

            if context.space_data.type == "VIEW_3D":

                # Assign Model name to CuttingTarget property :
                self.CuttingTarget = CuttingTarget = bpy.context.view_layer.objects.active
                bpy.context.scene.BDENTAL_4D_Props.CuttingTargetNameProp = (
                    CuttingTarget.name
                )

                bpy.ops.object.mode_set(mode="OBJECT")
                # bpy.ops.object.hide_view_clear()
                bpy.ops.object.select_all(action="DESELECT")

                # for obj in bpy.data.objects:
                #     if "CuttingCurve" in obj.name:
                #         obj.select_set(True)
                #         bpy.ops.object.delete(use_global=False, confirm=False)

                bpy.ops.object.select_all(action="DESELECT")
                CuttingTarget.select_set(True)
                bpy.context.view_layer.objects.active = CuttingTarget
                # Hide everything but model and cutters:
                CuttingTarget.hide_set(False)
                Cutters_Coll = bpy.data.collections.get("BDENTAL-4D Cutters")
                if Cutters_Coll :
                    Cutters_Objects = Cutters_Coll.objects
                    if Cutters_Objects :
                        [obj.hide_set(False) for obj in Cutters_Objects]
                

                self.CurveCutter = CuttingCurveAdd2()

                context.window_manager.modal_handler_add(self)

                return {"RUNNING_MODAL"}

            else:

                self.report({"WARNING"}, "Active space must be a View3d")

                return {"CANCELLED"}


################################################################################
class BDENTAL_4D_OT_CurveCutter2_ShortPath(bpy.types.Operator):
    " Shortpath Curve Cutting tool"

    bl_idname = "bdental4d.curvecutter2_shortpath"
    bl_label = "ShortPath"

    Resolution: IntProperty(
        name="Cut Resolution",
        description="Cutting curve Resolution",
        default=3,
    )

    def execute(self, context):

        start = Tcounter()
        BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props
        ###########################################################################
        if bpy.context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        
        # Get CuttingTarget :
        CuttingTargetName = BDENTAL_4D_Props.CuttingTargetNameProp
        CuttingTarget = bpy.data.objects[CuttingTargetName]
        CuttingTarget.hide_select = False
        # delete old vertex groups :
        CuttingTarget.vertex_groups.clear()
        # Get CurveCutter :
        CurveCuttersList = [obj for obj in bpy.data.objects if 'BDENTAL4D_Curve_Cut2' in obj.name]

        CurveMeshesList = []
        for CurveCutter in CurveCuttersList:
            CurveCutter.hide_select = False
            bpy.ops.object.select_all(action="DESELECT")
            CurveCutter.select_set(True)
            bpy.context.view_layer.objects.active = CurveCutter

            HookModifiers = [mod.name for mod in CurveCutter.modifiers if 'Hook' in mod.name]
            for mod in HookModifiers :
                bpy.ops.object.modifier_apply(modifier=mod)

            CurveCutter.data.bevel_depth = 0
            CurveCutter.data.resolution_u = self.Resolution
            bpy.ops.object.modifier_apply(modifier="Shrinkwrap")

            bpy.ops.object.convert(target="MESH")
            CurveCutter = context.object
            bpy.ops.object.modifier_add(type="SHRINKWRAP")
            CurveCutter.modifiers["Shrinkwrap"].target = CuttingTarget
            bpy.ops.object.convert(target='MESH')

            CurveMesh = context.object
            CurveMeshesList.append(CurveMesh)

        bpy.ops.object.select_all(action="DESELECT")
        CuttingTarget.select_set(True)
        bpy.context.view_layer.objects.active = CuttingTarget
        me = CuttingTarget.data
        # initiate a KDTree :
        size = len(me.vertices)
        kd = kdtree.KDTree(size)

        for v_id, v in enumerate(me.vertices):
            kd.insert(v.co, v_id)

        kd.balance()
        Loop = []
        for CurveCutter in CurveMeshesList:
            
            CutterCoList = [CuttingTarget.matrix_world.inverted() @ CurveCutter.matrix_world @ v.co for v in CurveCutter.data.vertices]
            Closest_VIDs = [ kd.find(CutterCoList[i])[1] for i in range(len(CutterCoList))]
            print('Get closest verts list done')
            if BDENTAL_4D_Props.CurveCutCloseMode == 'Close Curve' :
                CloseState = True
            else :
                CloseState = False
            CutLine = ShortestPath(CuttingTarget, Closest_VIDs, close=CloseState)
            Loop.extend(CutLine)

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action='DESELECT')

        bpy.ops.object.mode_set(mode="OBJECT")
        for Id in Loop :
            me.vertices[Id].select = True

        print("Cut Line selected...")

        # bpy.ops.object.mode_set(mode="EDIT")
        # bpy.ops.mesh.loop_to_region()
        # bpy.ops.mesh.select_more()
        # bpy.ops.mesh.subdivide(number_cuts=2)
        # bpy.ops.mesh.select_all(action='DESELECT')

        # dg = context.evaluated_depsgraph_get()
        # me = me.evaluated_get(dg)
        # # initiate a KDTree :
        # size = len(me.vertices)
        # kd = kdtree.KDTree(size)

        # for v_id, v in enumerate(me.vertices):
        #     kd.insert(v.co, v_id)

        # kd.balance()
        # Loop = []
        # for CurveCutter in CurveMeshesList:
            
        #     CutterCoList = [CuttingTarget.matrix_world.inverted() @ CurveCutter.matrix_world @ v.co for v in CurveCutter.data.vertices]
        #     Closest_VIDs = [ kd.find(CutterCoList[i])[1] for i in range(len(CutterCoList))]
        #     print('Get closest verts list done')
        #     if BDENTAL_4D_Props.CurveCutCloseMode == 'Close Curve' :
        #         CloseState = True
        #     else :
        #         CloseState = False
        #     CutLine = ShortestPath(CuttingTarget, Closest_VIDs, close=CloseState)
        #     Loop.extend(CutLine)

        # bpy.ops.object.mode_set(mode="OBJECT")
        # for Id in Loop :
        #     me.vertices[Id].select = True

        bpy.ops.object.mode_set(mode="EDIT")
        vg = CuttingTarget.vertex_groups.new(name="intersect_vgroup")
        bpy.ops.object.vertex_group_assign()

        print("Shrinkwrap Modifier...")
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        for CurveCutter in CurveMeshesList:
            CurveCutter.select_set(True)
            bpy.context.view_layer.objects.active = CurveCutter
        
        if len(CurveMeshesList)>1:
            bpy.ops.object.join()

        CurveCutter = context.object
        print("CurveCutter",CurveCutter)
        print("CuttingTarget",CuttingTarget)
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        CuttingTarget.select_set(True)
        bpy.context.view_layer.objects.active = CuttingTarget

        bpy.ops.object.modifier_add(type="SHRINKWRAP")
        CuttingTarget.modifiers["Shrinkwrap"].wrap_method = "NEAREST_VERTEX"
        CuttingTarget.modifiers["Shrinkwrap"].vertex_group = vg.name
        CuttingTarget.modifiers["Shrinkwrap"].target = CurveCutter
        bpy.ops.object.modifier_apply(modifier="Shrinkwrap")

        print("Relax Cut Line...")
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.looptools_relax(
            input="selected", interpolation="cubic", iterations="3", regular=True
        )

        print("Split Cut Line...")
        # bpy.ops.object.mode_set(mode="OBJECT")
        # Hiden = [v for v in me.vertices if v.select]
        # bpy.ops.object.mode_set(mode="EDIT")
        # bpy.ops.mesh.hide(unselected=False)
        # bpy.ops.object.mode_set(mode="OBJECT")
        # Visible = [v for v in me.vertices if not v in Hiden]
        # Visible[0].select = True
        # bpy.ops.object.mode_set(mode="EDIT")
        # bpy.ops.mesh.select_linked(delimit=set())
        # bpy.ops.mesh.reveal()
        # bpy.ops.mesh.separate(type="SELECTED")

        # Split :
        SplitSeparator(CuttingTarget=CuttingTarget)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")
        

        print("Remove Cutter tool...")
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        for obj in context.visible_objects:
            if (
                obj.type == "MESH"
                and len(obj.data.polygons) <= 10
            ):
                bpy.data.objects.remove(obj)
        col = bpy.data.collections['BDENTAL-4D Cutters']
        for obj in col.objects :
            bpy.data.objects.remove(obj)
        bpy.data.collections.remove(col)

        finish = Tcounter()
        print("finished in : ", finish - start, "secondes")
        return {"FINISHED"}

    def invoke(self, context, event):

        wm = context.window_manager
        return wm.invoke_props_dialog(self)



# CurveCutter_03
class BDENTAL_4D_OT_CurveCutterAdd3(bpy.types.Operator):
    """ description of this Operator """

    bl_idname = "bdental4d.curvecutteradd3"
    bl_label = "CURVE CUTTER ADD"
    bl_options = {"REGISTER", "UNDO"}

    def modal(self, context, event):

        BDENTAL_4D_Props = context.scene.BDENTAL_4D_Props

        if not event.type in {
            "DEL",
            "LEFTMOUSE",
            "RET",
            "ESC",
        }:
            # allow navigation

            return {"PASS_THROUGH"}

        elif event.type == ("DEL"):
            if event.value == ("PRESS"):

                DeleteLastCurvePoint()

            return {"RUNNING_MODAL"}

        elif event.type == ("LEFTMOUSE"):

            if event.value == ("PRESS"):

                return {"PASS_THROUGH"}

            if event.value == ("RELEASE"):

                ExtrudeCurvePointToCursor(context, event)

        elif event.type == "RET":

            if event.value == ("PRESS"):
                CurveCutterName = BDENTAL_4D_Props.CurveCutterNameProp
                CurveCutter = bpy.data.objects[CurveCutterName]
                CurveCutter.select_set(True)
                bpy.context.view_layer.objects.active = CurveCutter

                bpy.ops.object.mode_set(mode="OBJECT")

                if BDENTAL_4D_Props.CurveCutCloseMode == "Close Curve":
                    bpy.ops.object.mode_set(mode="EDIT")
                    bpy.ops.curve.cyclic_toggle()
                    bpy.ops.object.mode_set(mode="OBJECT")

                # bpy.ops.object.modifier_apply(apply_as="DATA", modifier="Shrinkwrap")

                bpy.ops.wm.tool_set_by_id(name="builtin.select")
                # bpy.context.scene.tool_settings.use_snap = False
                bpy.context.space_data.overlay.show_outline_selected = True

                bezier_points = CurveCutter.data.splines[0].bezier_points[:]
                for i in range(len(bezier_points)):
                    AddCurveSphere(
                        Name=f"Hook_{i}",
                        Curve=CurveCutter,
                        i=i,
                        CollName="BDENTAL-4D Cutters",
                    )
                bpy.context.space_data.overlay.show_relationship_lines = False

                return {"FINISHED"}

        elif event.type == ("ESC"):

            if event.value == ("PRESS"):

                CurveCutterName = bpy.context.scene.BDENTAL_4D_Props.CurveCutterNameProp
                CurveCutter = bpy.data.objects[CurveCutterName]
                bpy.ops.object.mode_set(mode="OBJECT")

                bpy.ops.object.select_all(action="DESELECT")
                CurveCutter.select_set(True)
                bpy.context.view_layer.objects.active = CurveCutter
                bpy.ops.object.delete(use_global=False, confirm=False)

                CuttingTargetName = context.scene.BDENTAL_4D_Props.CuttingTargetNameProp
                CuttingTarget = bpy.data.objects[CuttingTargetName]

                bpy.ops.object.select_all(action="DESELECT")
                CuttingTarget.select_set(True)
                bpy.context.view_layer.objects.active = CuttingTarget

                bpy.ops.wm.tool_set_by_id(name="builtin.select")
                bpy.context.scene.tool_settings.use_snap = True
                bpy.context.space_data.overlay.show_outline_selected = True
                bpy.context.scene.tool_settings.snap_target = "CENTER"
                bpy.context.scene.tool_settings.snap_elements = {"FACE"}
                bpy.context.space_data.overlay.show_relationship_lines = False

                return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):

        if bpy.context.selected_objects == []:

            message = [" Please select the target object !"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:

            if context.space_data.type == "VIEW_3D":

                # Assign Model name to CuttingTarget property :
                CuttingTarget = bpy.context.view_layer.objects.active
                bpy.context.scene.BDENTAL_4D_Props.CuttingTargetNameProp = (
                    CuttingTarget.name
                )

                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.hide_view_clear()
                bpy.ops.object.select_all(action="DESELECT")

                # for obj in bpy.data.objects:
                #     if "CuttingCurve" in obj.name:
                #         obj.select_set(True)
                #         bpy.ops.object.delete(use_global=False, confirm=False)

                bpy.ops.object.select_all(action="DESELECT")
                CuttingTarget.select_set(True)
                bpy.context.view_layer.objects.active = CuttingTarget
                # Hide everything but model :
                bpy.ops.object.hide_view_set(unselected=True)

                CuttingCurveAdd2()

                context.window_manager.modal_handler_add(self)

                return {"RUNNING_MODAL"}

            else:

                self.report({"WARNING"}, "Active space must be a View3d")

                return {"CANCELLED"}


class BDENTAL_4D_OT_CurveCutterCut3(bpy.types.Operator):
    " Performe Curve Cutting Operation"

    bl_idname = "bdental4d.curvecuttercut3"
    bl_label = "CURVE CUTTER CUT"

    def execute(self, context):

        # Get CuttingTarget :
        CuttingTargetName = bpy.context.scene.BDENTAL_4D_Props.CuttingTargetNameProp
        CuttingTarget = bpy.data.objects[CuttingTargetName]

        # Get CurveCutter :
        bpy.ops.object.select_all(action="DESELECT")

        CurveCuttersList = [
            obj
            for obj in context.visible_objects
            if obj.type == "CURVE" and obj.name.startswith("BDENTAL4D_Curve_Cut")
        ]

        if not CurveCuttersList:

            message = [
                " Can't find curve Cutters ",
                "Please ensure curve Cutters are not hiden !",
            ]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        if CurveCuttersList:
            CurveMeshesList = []
            for CurveCutter in CurveCuttersList:
                bpy.ops.object.select_all(action="DESELECT")
                CurveCutter.select_set(True)
                bpy.context.view_layer.objects.active = CurveCutter

                # remove material :
                for _ in CurveCutter.material_slots:
                    bpy.ops.object.material_slot_remove()

                # Change CurveCutter setting   :
                CurveCutter.data.bevel_depth = 0
                CurveCutter.data.resolution_u = 6

                # Add shrinkwrap modif outside :
                bpy.ops.object.modifier_add(type="SHRINKWRAP")
                CurveCutter.modifiers["Shrinkwrap"].use_apply_on_spline = True
                CurveCutter.modifiers["Shrinkwrap"].target = CuttingTarget
                CurveCutter.modifiers["Shrinkwrap"].offset = 0.5
                CurveCutter.modifiers["Shrinkwrap"].wrap_mode = "OUTSIDE"

                # duplicate curve :
                bpy.ops.object.duplicate_move()
                CurveCutterDupli = context.object
                CurveCutterDupli.modifiers["Shrinkwrap"].wrap_mode = "INSIDE"
                CurveCutterDupli.modifiers["Shrinkwrap"].offset = 0.8

                IntOut = []
                for obj in [CurveCutter, CurveCutterDupli]:
                    # convert CurveCutter to mesh :
                    bpy.ops.object.mode_set(mode="OBJECT")
                    bpy.ops.object.select_all(action="DESELECT")
                    obj.select_set(True)
                    bpy.context.view_layer.objects.active = obj
                    bpy.ops.object.convert(target="MESH")
                    CurveMesh = context.object
                    IntOut.append(CurveMesh)

                bpy.ops.object.select_all(action="DESELECT")
                for obj in IntOut:
                    obj.select_set(True)
                    bpy.context.view_layer.objects.active = obj
                bpy.ops.object.join()
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.bridge_edge_loops()
                bpy.ops.object.mode_set(mode="OBJECT")
                CurveMeshesList.append(context.object)

            bpy.ops.object.select_all(action="DESELECT")
            for obj in CurveMeshesList:
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj

            if len(CurveMeshesList) > 1:
                bpy.ops.object.join()

            CurveCutter = context.object

            CurveCutter.select_set(True)
            bpy.context.view_layer.objects.active = CurveCutter

            bpy.context.scene.tool_settings.use_snap = False
            bpy.ops.view3d.snap_cursor_to_center()

            # # Make vertex group :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.context.tool_settings.mesh_select_mode = (True, False, False)
            bpy.ops.mesh.select_all(action="SELECT")
            curve_vgroup = CurveCutter.vertex_groups.new(name="curve_vgroup")
            bpy.ops.object.vertex_group_assign()
            bpy.ops.object.mode_set(mode="OBJECT")

            # select CuttingTarget :
            bpy.ops.object.select_all(action="DESELECT")
            CuttingTarget.select_set(True)
            bpy.context.view_layer.objects.active = CuttingTarget

            # delete old vertex groups :
            CuttingTarget.vertex_groups.clear()

            # deselect all vertices :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.mesh.select_all(action="DESELECT")
            bpy.ops.object.mode_set(mode="OBJECT")

            ###############################################################

            # Join CurveCutter to CuttingTarget :
            CurveCutter.select_set(True)
            bpy.ops.object.join()
            bpy.ops.object.hide_view_set(unselected=True)

            # intersect make vertex group :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.intersect()

            intersect_vgroup = CuttingTarget.vertex_groups.new(name="intersect_vgroup")
            CuttingTarget.vertex_groups.active_index = intersect_vgroup.index
            bpy.ops.object.vertex_group_assign()

            # # delete curve_vgroup :
            # bpy.ops.object.mode_set(mode="EDIT")
            # bpy.ops.mesh.select_all(action="DESELECT")
            # curve_vgroup = CuttingTarget.vertex_groups["curve_vgroup"]

            # CuttingTarget.vertex_groups.active_index = curve_vgroup.index
            # bpy.ops.object.vertex_group_select()
            # bpy.ops.mesh.delete(type="FACE")

            # bpy.ops.ed.undo_push()
            # # 1st methode :
            # SplitSeparator(CuttingTarget=CuttingTarget)

            # for obj in context.visible_objects:
            #     if len(obj.data.polygons) <= 10:
            #         bpy.data.objects.remove(obj)
            # for obj in context.visible_objects:
            #     if obj.name.startswith("Hook"):
            #         bpy.data.objects.remove(obj)

            # print("Cutting done with first method")

            return {"FINISHED"}


#######################################################################################
# Square cut modal operator :


class BDENTAL_4D_OT_square_cut(bpy.types.Operator):
    """Square Cutting Tool add"""

    bl_idname = "bdental4d.square_cut"
    bl_label = "Square Cut"
    bl_options = {"REGISTER", "UNDO"}

    def modal(self, context, event):

        if event.type == "RET":
            if event.value == ("PRESS"):

                add_square_cutter(context)

            return {"FINISHED"}

        elif event.type == ("ESC"):

            return {"CANCELLED"}

        else:

            # allow navigation
            return {"PASS_THROUGH"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):

        if bpy.context.selected_objects == []:

            message = [" Please select the target object !"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:

            if context.space_data.type == "VIEW_3D":

                CuttingTarget = context.scene.BDENTAL_4D_Props.CuttingTargetNameProp

                # Hide everything but model :

                bpy.ops.object.mode_set(mode="OBJECT")

                Model = bpy.context.view_layer.objects.active
                bpy.ops.object.select_all(action="DESELECT")
                Model.select_set(True)

                CuttingTarget = Model.name

                bpy.ops.object.hide_view_set(unselected=True)

                bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")
                bpy.context.scene.tool_settings.use_snap = False

                message = [
                    " Please align Model to the Cutting View and click 'ENTER' !"
                ]
                ShowMessageBox(message=message, icon="COLORSET_02_VEC")

                context.window_manager.modal_handler_add(self)

                return {"RUNNING_MODAL"}

            else:

                self.report({"WARNING"}, "Active space must be a View3d")

                return {"CANCELLED"}


#######################################################################################
# Square cut confirm operator :


class BDENTAL_4D_OT_square_cut_confirm(bpy.types.Operator):
    """confirm Square Cut operation"""

    bl_idname = "bdental4d.square_cut_confirm"
    bl_label = "Tirm"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        if bpy.context.selected_objects == []:

            message = [" Please select the target object !"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:
            try:
                cutting_mode = context.scene.BDENTAL_4D_Props.cutting_mode

                bpy.context.tool_settings.mesh_select_mode = (True, False, False)
                bpy.ops.wm.tool_set_by_id(name="builtin.select")
                bpy.ops.object.mode_set(mode="OBJECT")
                frame = bpy.data.objects["my_frame_cutter"]

                bpy.ops.object.select_all(action="DESELECT")
                frame.select_set(True)
                bpy.context.view_layer.objects.active = frame
                bpy.ops.object.select_all(action="INVERT")
                Model = bpy.context.selected_objects[0]
                bpy.context.view_layer.objects.active = Model

                # Make Model normals consitent :

                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.normals_make_consistent(inside=False)
                bpy.ops.mesh.select_all(action="DESELECT")
                bpy.ops.object.mode_set(mode="OBJECT")

                # ....Add undo history point...:
                bpy.ops.ed.undo_push()

                # Add Boolean Modifier :
                bpy.ops.object.select_all(action="DESELECT")
                Model.select_set(True)
                bpy.context.view_layer.objects.active = Model

                bpy.ops.object.modifier_add(type="BOOLEAN")
                bpy.context.object.modifiers["Boolean"].show_viewport = False
                bpy.context.object.modifiers["Boolean"].operation = "DIFFERENCE"
                bpy.context.object.modifiers["Boolean"].object = frame

                # Apply boolean modifier :
                if cutting_mode == "Cut inner":
                    bpy.ops.object.modifier_apply(modifier="Boolean")

                if cutting_mode == "Keep inner":
                    bpy.context.object.modifiers["Boolean"].operation = "INTERSECT"
                    bpy.ops.object.modifier_apply(modifier="Boolean")

                # Delete resulting loose geometry :

                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.delete_loose()
                bpy.ops.mesh.select_all(action="DESELECT")
                bpy.ops.object.mode_set(mode="OBJECT")

                bpy.ops.object.select_all(action="DESELECT")
                frame.select_set(True)
                bpy.context.view_layer.objects.active = frame

            except Exception:
                pass

            return {"FINISHED"}


#######################################################################################
# Square cut exit operator :


class BDENTAL_4D_OT_square_cut_exit(bpy.types.Operator):
    """Square Cutting Tool Exit"""

    bl_idname = "bdental4d.square_cut_exit"
    bl_label = "Exit"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        # Delete frame :
        try:

            frame = bpy.data.objects["my_frame_cutter"]
            bpy.ops.object.select_all(action="DESELECT")
            frame.select_set(True)

            bpy.ops.object.select_all(action="INVERT")
            Model = bpy.context.selected_objects[0]

            bpy.ops.object.select_all(action="DESELECT")
            frame.select_set(True)
            bpy.context.view_layer.objects.active = frame

            bpy.ops.object.delete(use_global=False, confirm=False)

            bpy.ops.object.select_all(action="DESELECT")
            Model.select_set(True)
            bpy.context.view_layer.objects.active = Model

        except Exception:
            pass

        return {"FINISHED"}


class BDENTAL_4D_OT_PaintArea(bpy.types.Operator):
    """ Vertex paint area context toggle """

    bl_idname = "bdental4d.paintarea_toggle"
    bl_label = "PAINT AREA"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        ActiveObj = context.active_object
        if not ActiveObj:

            message = [" Please select the target object !"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:

            condition = ActiveObj.type == "MESH" and ActiveObj.select_get() == True

            if not condition:

                message = [" Please select the target object !"]
                ShowMessageBox(message=message, icon="COLORSET_02_VEC")

                return {"CANCELLED"}

            else:

                Override, area3D, space3D = CtxOverride(context)
                bpy.ops.object.mode_set(mode="VERTEX_PAINT")
                bpy.ops.wm.tool_set_by_id(name="builtin_brush.Draw")

                DrawBrush = bpy.data.brushes.get("Draw")
                DrawBrush.blend = "MIX"
                DrawBrush.color = (0.0, 1.0, 0.0)
                DrawBrush.strength = 1.0
                DrawBrush.use_frontface = True
                DrawBrush.use_alpha = True
                DrawBrush.stroke_method = "SPACE"
                DrawBrush.curve_preset = "CUSTOM"
                DrawBrush.cursor_color_add = (0.0, 0.0, 1.0, 0.9)
                DrawBrush.use_cursor_overlay = True

                bpy.context.tool_settings.vertex_paint.tool_slots[0].brush = DrawBrush

                for vg in ActiveObj.vertex_groups:
                    ActiveObj.vertex_groups.remove(vg)

                for VC in ActiveObj.data.vertex_colors:
                    ActiveObj.data.vertex_colors.remove(VC)

                ActiveObj.data.vertex_colors.new(name="BDENTAL_4D_PaintCutter_VC")

                return {"FINISHED"}


class BDENTAL_4D_OT_PaintAreaPlus(bpy.types.Operator):
    """ Vertex paint area Paint Plus toggle """

    bl_idname = "bdental4d.paintarea_plus"
    bl_label = "PLUS"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        if not context.mode == "PAINT_VERTEX":

            message = [
                " Please select the target object ",
                "and activate Vertex Paint mode !",
            ]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:
            Override, area3D, space3D = CtxOverride(context)
            bpy.ops.wm.tool_set_by_id(name="builtin_brush.Draw")
            DrawBrush = bpy.data.brushes.get("Draw")
            context.tool_settings.vertex_paint.tool_slots[0].brush = DrawBrush
            DrawBrush.blend = "MIX"
            DrawBrush.color = (0.0, 1.0, 0.0)
            DrawBrush.strength = 1.0
            DrawBrush.use_frontface = True
            DrawBrush.use_alpha = True
            DrawBrush.stroke_method = "SPACE"
            DrawBrush.curve_preset = "CUSTOM"
            DrawBrush.cursor_color_add = (0.0, 0.0, 1.0, 0.9)
            DrawBrush.use_cursor_overlay = True
            space3D.show_region_header = False
            space3D.show_region_header = True

            return {"FINISHED"}


class BDENTAL_4D_OT_PaintAreaMinus(bpy.types.Operator):
    """ Vertex paint area Paint Minus toggle """

    bl_idname = "bdental4d.paintarea_minus"
    bl_label = "MINUS"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        if not context.mode == "PAINT_VERTEX":

            message = [
                " Please select the target object ",
                "and activate Vertex Paint mode !",
            ]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:
            Override, area3D, space3D = CtxOverride(context)
            bpy.ops.wm.tool_set_by_id(name="builtin_brush.Draw")
            LightenBrush = bpy.data.brushes.get("Lighten")
            context.tool_settings.vertex_paint.tool_slots[0].brush = LightenBrush
            LightenBrush.blend = "MIX"
            LightenBrush.color = (1.0, 1.0, 1.0)
            LightenBrush.strength = 1.0
            LightenBrush.use_frontface = True
            LightenBrush.use_alpha = True
            LightenBrush.stroke_method = "SPACE"
            LightenBrush.curve_preset = "CUSTOM"
            LightenBrush.cursor_color_add = (1, 0.0, 0.0, 0.9)
            LightenBrush.use_cursor_overlay = True
            space3D.show_region_header = False
            space3D.show_region_header = True

            return {"FINISHED"}


class BDENTAL_4D_OT_PaintCut(bpy.types.Operator):
    """ Vertex paint Cut """

    bl_idname = "bdental4d.paint_cut"
    bl_label = "CUT"

    Cut_Modes_List = ["Cut", "Make Copy (Shell)", "Remove Painted", "Keep Painted"]
    items = []
    for i in range(len(Cut_Modes_List)):
        item = (str(Cut_Modes_List[i]), str(Cut_Modes_List[i]), str(""), int(i))
        items.append(item)

    Cut_Mode_Prop: EnumProperty(
        name="Cut Mode", items=items, description="Cut Mode", default="Cut"
    )

    def execute(self, context):

        VertexPaintCut(mode=self.Cut_Mode_Prop)
        bpy.ops.ed.undo_push(message="BDENTAL_4D Paint Cutter")

        return {"FINISHED"}

    def invoke(self, context, event):

        if not context.mode == "PAINT_VERTEX":

            message = [
                " Please select the target object ",
                "and activate Vertex Paint mode !",
            ]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:

            wm = context.window_manager
            return wm.invoke_props_dialog(self)
###########################################################################
# DSD Camera
###########################################################################
#---------------------------------------------------------------
# 3x4 P matrix from Blender camera
#---------------------------------------------------------------

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in 
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
def get_calibration_matrix_K_from_blender(DsdCam):
    f_in_mm = DsdCam.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = DsdCam.sensor_width
    sensor_height_in_mm = DsdCam.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (DsdCam.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal), 
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm
    

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels

    K = np.array(
        ((alpha_u, skew,    u_0),
        (    0  , alpha_v, v_0),
        (    0  , 0,        1 )))
    return K

# Returns camera rotation and translation matrices from Blender.
# 
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates 
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(DsdCam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = DsdCam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = DsdCam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*DsdCam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    # NOTE: Use * instead of @ here for older versions of Blender
    # TODO: detect Blender version
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
         ))
    return RT

def get_3x4_P_matrix_from_blender(DsdCam):
    K = get_calibration_matrix_K_from_blender(DsdCam.data)
    RT = get_3x4_RT_matrix_from_blender(DsdCam)
    return K@RT, K, RT

# ----------------------------------------------------------
# Alternate 3D coordinates to 2D pixel coordinate projection code
# adapted from https://blender.stackexchange.com/questions/882/how-to-find-image-coordinates-of-the-rendered-vertex?lq=1
# to have the y axes pointing up and origin at the top-left corner
def project_by_object_utils(cam, point):
    scene = bpy.context.scene
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, point)
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
            int(scene.render.resolution_x * render_scale),
            int(scene.render.resolution_y * render_scale),
            )
    return Vector((co_2d.x * render_size[0], render_size[1] - co_2d.y * render_size[1]))


#     p1 = P @ e1
#     p1 /= p1[2]
#     print("Projected e1")
#     print(p1)
#     print("proj by object_utils")
#     print(project_by_object_utils(cam, Vector(e1[0:3])))


def CamIntrisics(CalibFile):
    with open(CalibFile, "rb") as rf:
        (K, distCoeffs, _, _) = pickle.load(rf)
    fx, fy= K[0,0], K[1,1]
    return fx, fy, K, distCoeffs

def Undistort(DistImage, K, distCoeffs) :
    img = cv2.imread(DistImage)
    h,  w = img.shape[:2]
    if w<h:
        fx,fy,cx,cy = K[0,0],K[1,1],K[0,2], K[1,2]
        K=np.array([[fy,0,cy],[0,fx,cx],[0,0,1]],dtype=np.float32)

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, distCoeffs, (w,h), 1, (w,h))
    # undistort
    UndistImage = cv2.undistort(img, K, distCoeffs, None, newcameramtx)
    # crop the image
    # x, y, w, h = roi
    # UndistImage = UndistImage[y:y+h, x:x+w]
    Split = split(DistImage)
    UndistImagePath = join(Split[0],f"Undistorted_{Split[1]}")
    cv2.imwrite(UndistImagePath, UndistImage)
    
    return UndistImagePath
    
def DsdCam_from_CalibMatrix(fx, fy, cx, cy):
    if cx>cy :
        print('Horizontal mode')
        sensor_width_in_mm = fy*cx / (fx*cy)
        sensor_height_in_mm = 1  # doesn't matter
        s_u = cx*2 / sensor_width_in_mm
        f_in_mm = fx / s_u
    if cx<cy :
        print('Vertical mode')
        sensor_width_in_mm = fy*cy / (fx*cx)
        sensor_height_in_mm = 1  # doesn't matter
        s_u = cy*2 / sensor_width_in_mm
        f_in_mm = fx / s_u
    
    return sensor_width_in_mm,sensor_height_in_mm, f_in_mm

def Focal_lengh_To_K(f_in_mm,w,h):
    if w>h:
        cx,cy=w/2, h/2
        fx=fy=f_in_mm*h
        K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=np.float32)
        sensor_width_in_mm = fy*cx / (fx*cy)
    if w<h:
        cx,cy=w/2, h/2
        fx=fy=f_in_mm*w
        K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=np.float32)
        sensor_width_in_mm = fy*cy / (fx*cx)
    
    return K,sensor_width_in_mm,1


def DsdCam_Orientation(ObjPoints3D, Undistorted_Image_Points2D, K, cx, cy):

    K[0,2] = cx
    K[1,2] = cy
    distCoeffs = np.array([0.0,0.0,0.0,0.0,0.0])
    ret, rvec, tvec = cv2.solvePnP(ObjPoints3D, Undistorted_Image_Points2D, K, distCoeffs)

    mat = mathutils.Matrix(
                            [[1.0, 0.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]]    
                                                    )

    ViewCam_RotM = Matrix(cv2.Rodrigues(rvec)[0]) 
    ViewCam_Matrix = ViewCam_RotM.to_4x4()                   
    ViewCam_Matrix.translation = Vector(tvec)
    Cam_Matrix = ViewCam_Matrix.inverted() @ mat
    
    return Cam_Matrix

class BDENTAL_4D_OT_XrayToggle(bpy.types.Operator):
    """  """

    bl_idname = "bdental4d.xray_toggle"
    bl_label = "2D Image to 3D Matching"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        bpy.ops.view3d.toggle_xray()

        return {"FINISHED"}

class BDENTAL_4D_OT_Matching2D3D(bpy.types.Operator):
    """  """

    bl_idname = "bdental4d.matching_2d3d"
    bl_label = "2D Image to 3D Matching"
    bl_options = {"REGISTER", "UNDO"}

    MP2D_List = []
    MP3D_List = []
    MP_All_List = []

    counter2D, counter3D = 1, 1
    color2D = (1,0,1,1)
    color3D = (0,1,0,1)


    def MouseActionZone(self, context, event):
        if event.mouse_x >= self.Right_A3D.x :
            return '2D'
        else :
            return '3D'

    def modal(self, context, event):

        if not event.type in {
            "DEL",
            "LEFTMOUSE",
            "RET",
            "ESC",
        }:
            # allow navigation

            return {"PASS_THROUGH"}

        elif event.type == ("DEL"):
            
            if event.value == ("PRESS"):
                
                if self.MP_All_List :
                    obj = self.MP_All_List.pop()
                    bpy.data.objects.remove(obj)
                    
                    if obj in self.MP2D_List :
                        self.counter2D-=1
                        self.MP2D_List.pop()
                        self.Matchs['2D'].pop()
                    if obj in self.MP3D_List :
                        self.counter3D-=1
                        self.MP3D_List.pop()
                        self.Matchs['3D'].pop()

            return {"RUNNING_MODAL"}

        elif event.type == ("LEFTMOUSE"):

            if event.value == ("PRESS"):

                bpy.ops.wm.tool_set_by_id(name="builtin.cursor")
                return {"PASS_THROUGH"}

            if event.value == ("RELEASE"):
                
                Cursor = context.scene.cursor.location
                CursorLocal = self.Mtx @ Cursor
                u = (CursorLocal[0] + self.Dims[0]/2)/self.Dims[0]
                v = (CursorLocal[1] + self.Dims[1]/2)/self.Dims[1]
                w = CursorLocal[2]
                
                e = 0.0001
                
                MAZ = self.MouseActionZone(context, event)
                
                if MAZ == '2D' :
                    if not 0<=u+e<=1 or not 0<=v+e<=1 or w > e :
                        self.report({'INFO'}, f"Outside the Image : (u : {u} , v : {v}, w : {w})")
                    else :
                        name = f"MP2D_{self.counter2D}"
                        MP2D = AddMarkupPoint(name, self.color2D, Cursor, Diameter=0.01,CollName='CAM_DSD', show_name=False)
                        self.MP2D_List.append(MP2D)
                        self.MP_All_List.append(MP2D)
                        
                        bpy.ops.object.select_all(action="DESELECT")

                        size_x, size_y = self.size
                    
                        px, py = int((size_x -1)*u), int((size_y -1)*(1-v))

                        self.Matchs['2D'].append([px, py])
                        self.counter2D+=1

                if MAZ == '3D' :
                    name = f"MP3D_{self.counter3D}"
                    MP3D = AddMarkupPoint(name, self.color3D, Cursor, Diameter=1, show_name=False)
                    self.MP3D_List.append(MP3D)
                    self.MP_All_List.append(MP3D)
                    
                    bpy.ops.object.select_all(action="DESELECT")
                    self.Matchs['3D'].append(list(MP3D.location))
                    self.counter3D+=1
                        
                bpy.ops.wm.tool_set_by_id(name="builtin.select")
                self.ImagePlane.select_set(True)
                context.view_layer.objects.active = self.ImagePlane

        elif event.type == "RET":
            bpy.ops.object.hide_collection(self.Right_override, collection_index=1,toggle=True)
            Points = [p for p in bpy.data.objects if 'MP2D' in p.name or 'MP3D' in p.name]
            Points2D = [p for p in bpy.data.objects if 'MP2D' in p.name]
            Points3D = [p for p in bpy.data.objects if 'MP3D' in p.name]
            if len(Points2D) != len(Points3D) :
                message = [
                "Number of Points is not equal !", "Please check and retry !",
                ]
                ShowMessageBox(message=message, icon="COLORSET_02_VEC")
                return {"RUNNING_MODAL"}

            else :
                if Points :
                    for p in Points :
                        bpy.data.objects.remove(p)

                arr3D = np.array(self.Matchs['3D'],dtype=np.float32)
                arr2D = np.array(self.Matchs['2D'],dtype=np.float32)
                

                invMtx = self.Cam_obj.matrix_world.inverted().copy()
                Cam_Matrix = DsdCam_Orientation(arr3D, arr2D,  self.K, self.cx, self.cy)
                self.Cam_obj.matrix_world = Cam_Matrix
                self.ImagePlane.matrix_world = Cam_Matrix @ invMtx @ self.ImagePlane.matrix_world
                self.ImagePlane.hide_set(True)
                bpy.ops.view3d.toggle_xray(self.Right_override)
                self.Right_S3D.shading.xray_alpha = 0.1


                

                return {"FINISHED"}

        elif event.type == ("ESC"):
            Points = [p for p in bpy.data.objects if 'MP2D' in p.name or 'MP3D' in p.name]
            if Points :
                for obj in Points:
                    bpy.data.objects.remove(obj)

            col = bpy.data.collections.get('Matching Points')
            if col :
                bpy.data.collections.remove(col)
            return {"CANCELLED"}

        return {"RUNNING_MODAL"}

    def invoke(self, context, event):

        BDENTAL_4D_Props = context.scene.BDENTAL_4D_Props
        ImagePath = BDENTAL_4D_Props.Back_ImageFile
        CalibFile = AbsPath(BDENTAL_4D_Props.DSD_CalibFile)

        if not exists(ImagePath):
            message = [
                "Please check Image path and retry !",
            ]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        self.Target = context.object
        if not self.Target:

            message = [" Please select the target object !"]
            ShowMessageBox(message=message, icon="COLORSET_02_VEC")

            return {"CANCELLED"}

        else:

            

            if context.space_data.type == "VIEW_3D":
                #######################################################
                if not CalibFile :
                
                    ImgName = os.path.split(ImagePath)[-1] or os.path.split(ImagePath)[-2]
                    self.Suffix = ImgName.split('.')[0]

                    ImageName = f"DSD_Image({self.Suffix})"
                    self.Dsd_Image = Image = bpy.data.images.get(ImageName) or bpy.data.images.load(ImagePath, check_existing=False)

                    Image.name = ImageName
                    Image.colorspace_settings.name = 'Non-Color'

                    #Add Camera :
                    bpy.ops.object.camera_add(location=(0, 0, 0), rotation=(pi, 0, 0), scale=(1, 1, 1))
                    self.Cam_obj = context.object
                    self.Cam_obj.name = f"DSD_Camera({self.Suffix})"
                    MoveToCollection(self.Cam_obj, 'CAM_DSD')
                    Cam = self.Cam_obj.data
                    Cam.name = self.Cam_obj.name
                    Cam.type = 'PERSP'
                    Cam.lens_unit = 'MILLIMETERS'
                    Cam.display_size = 10
                    Cam.show_background_images = True

                    # Make background Image :
                    self.Cam_obj.data.background_images.new()
                    bckg_Image = self.Cam_obj.data.background_images[0]
                    bckg_Image.image = self.Dsd_Image
                    bckg_Image.display_depth = 'BACK'
                    bckg_Image.alpha = 0.9

                    ######################################
                    # sensor_width_in_mm = Cam.sensor_width
                    # sensor_height_in_mm = Cam.sensor_height
                    # f_in_mm = Cam.lens
                    W, H = Image.size[:]
                    render = context.scene.render
                    render.resolution_percentage = 100
                    render.resolution_x = W
                    render.resolution_y = H

                    Cam.sensor_fit = 'AUTO'
                    f_in_mm = 3.80
                    # W, H = Image.size[:]
                    # self.K = get_calibration_matrix_K_from_blender(DsdCam=Cam)
                    self.K,sensor_width_in_mm,sensor_height_in_mm = Focal_lengh_To_K(f_in_mm,W,H)
                    self.cx, self.cy = W/2, H/2

                # #######################################################
                if CalibFile :
                    if not exists(CalibFile) :
                        message = [
                            "Please check Camera Calibration file and retry !",
                        ]
                        ShowMessageBox(message=message, icon="COLORSET_02_VEC")

                        return {"CANCELLED"}
                    
                    fx, fy, self.K, distCoeffs = CamIntrisics(CalibFile)

                    UndistImagePath = Undistort(ImagePath, self.K, distCoeffs)

                    ImgName = os.path.split(ImagePath)[-1] or os.path.split(ImagePath)[-2]
                    self.Suffix = ImgName.split('.')[0]

                    ImageName = f"DSD_Image({self.Suffix})"
                    self.Dsd_Image = Image = bpy.data.images.get(ImageName) or bpy.data.images.load(UndistImagePath, check_existing=False)
                    # self.Dsd_Image = Image = bpy.data.images.get(ImageName) or bpy.data.images.load(ImagePath, check_existing=False)

                    Image.name = ImageName
                    Image.colorspace_settings.name = 'Non-Color'

                    #Add Camera :
                    bpy.ops.object.camera_add(location=(0, 0, 0), rotation=(pi, 0, 0), scale=(1, 1, 1))
                    self.Cam_obj = context.object
                    self.Cam_obj.name = f"DSD_Camera({self.Suffix})"
                    MoveToCollection(self.Cam_obj, 'CAM_DSD')
                    Cam = self.Cam_obj.data
                    Cam.name = self.Cam_obj.name
                    Cam.type = 'PERSP'
                    Cam.lens_unit = 'MILLIMETERS'
                    Cam.display_size = 10
                    Cam.show_background_images = True

                    # Make background Image :
                    self.Cam_obj.data.background_images.new()
                    bckg_Image = self.Cam_obj.data.background_images[0]
                    bckg_Image.image = self.Dsd_Image
                    bckg_Image.display_depth = 'BACK'
                    bckg_Image.alpha = 0.9

                    ######################################
                    W, H = Image.size[:]
                    render = context.scene.render
                    render.resolution_percentage = 100
                    render.resolution_x = W
                    render.resolution_y = H

                    Cam.sensor_fit = 'AUTO'

                    self.cx, self.cy = W/2, H/2
                    
                    sensor_width_in_mm,sensor_height_in_mm, f_in_mm = DsdCam_from_CalibMatrix(fx, fy, self.cx, self.cy)

                # render = context.scene.render
                # render.resolution_percentage = 100
                # render.resolution_x = W
                # render.resolution_y = H

                # Cam.sensor_fit = 'AUTO'

                Cam.sensor_width  = sensor_width_in_mm
                Cam.sensor_height = sensor_height_in_mm
                Cam.lens = f_in_mm
                
                frame = Cam.view_frame()
                Cam_frame_World = [self.Cam_obj.matrix_world @ co for co in frame]
                Plane_loc = (Cam_frame_World[0] + Cam_frame_World[2])/2
                Plane_Dims = [W/max([W,H]), H/max([W,H]),0]
                # Plane_Dims = [1, H/W,0]

                bpy.ops.mesh.primitive_plane_add(location=Plane_loc, rotation=self.Cam_obj.rotation_euler)
                self.ImagePlane = bpy.context.object
                self.DSD_Coll = MoveToCollection(self.ImagePlane, 'BDENTAL_4D_DSD')
                self.ImagePlane.name = f"DSD_Plane_{self.Suffix}"
                self.ImagePlane.dimensions = Plane_Dims
                bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
                bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
                


                mat = bpy.data.materials.new(f"DSD_Mat_{self.Suffix}")
                mat.use_nodes = True
                node_tree = mat.node_tree
                nodes = node_tree.nodes
                links = node_tree.links

                for node in nodes:
                    if node.type != "OUTPUT_MATERIAL":
                        nodes.remove(node)

                TextureCoord = AddNode(nodes, type="ShaderNodeTexCoord", name="TextureCoord")
                ImageTexture = AddNode(nodes, type="ShaderNodeTexImage", name="Image Texture")

                ImageTexture.image = Image

                materialOutput = nodes["Material Output"]

                links.new(TextureCoord.outputs[0], ImageTexture.inputs[0])
                links.new(ImageTexture.outputs["Color"], materialOutput.inputs["Surface"])
                for slot in self.ImagePlane.material_slots:
                    bpy.ops.object.material_slot_remove()

                self.ImagePlane.active_material = mat

                mat.blend_method = "BLEND"
                context.space_data.shading.type = "SOLID"
                context.space_data.shading.color_type = "TEXTURE"
                context.space_data.shading.show_specular_highlight = False


                ##############################################################
                # Split area :
                WM = bpy.context.window_manager
                Window = WM.windows[-1]
                Screen = Window.screen

                bpy.ops.screen.area_split(direction="VERTICAL", factor=1 / 2)
                Areas3D = [
                    area for area in Screen.areas if area.type == "VIEW_3D"
                ]
                for area in Areas3D :
                    area.type = 'CONSOLE'
                    area.type = "VIEW_3D"

                    if area.x == 0 :
                        print('Area left found')
                        self.Left_A3D = Left_A3D =area
                        self.Left_S3D = Left_S3D = [
                                        space for space in Left_A3D.spaces if space.type == "VIEW_3D"
                                    ][0]
                        self.Left_R3D = Left_R3D =[
                                        reg for reg in Left_A3D.regions if reg.type == "WINDOW"
                                    ][0]
                        self.Left_override = Left_override = {'area':Left_A3D, 'space_data':Left_S3D, "region": Left_R3D}
                        Left_S3D.show_region_ui = False
                        Left_S3D.use_local_collections = True
                        
                    else :
                        print('Area right found')

                        self.Right_A3D = Right_A3D = area
                        self.Right_S3D = Right_S3D = [
                                        space for space in Right_A3D.spaces if space.type == "VIEW_3D"
                                    ][0]
                        self.Right_R3D = Right_R3D = [
                                        reg for reg in Right_A3D.regions if reg.type == "WINDOW"
                                    ][0]

                        self.Right_override = Right_override = {'area':Right_A3D, 'space_data':Right_S3D, "region": Right_R3D}
                        Right_S3D.show_region_ui = False
                        Right_S3D.use_local_collections = True
                        

                
                

                bpy.ops.view3d.view_camera(Right_override)
                bpy.ops.view3d.view_center_camera(Right_override)
                
                for i in range(2,len(bpy.data.collections)+1) :
                    bpy.ops.object.hide_collection(Left_override, collection_index=i,toggle=True)
                bpy.ops.object.hide_collection(Right_override, collection_index=1,toggle=True)
                #######################################################
                
                self.Mtx = self.ImagePlane.matrix_world.inverted()
                self.size = Image.size
                self.Dims = self.ImagePlane.dimensions
                self.Matchs = {'2D':[], '3D':[]}

                bpy.ops.wm.tool_set_by_id(name="builtin.select")
                self.ImagePlane.select_set(True)
                context.view_layer.objects.active = self.ImagePlane
                
                context.window_manager.modal_handler_add(self)

                return {"RUNNING_MODAL"}

            else:

                self.report({"WARNING"}, "Active space must be a View3d")

                return {"CANCELLED"}


#################################################################################################
# Registration :
#################################################################################################

classes = [
    BDENTAL_4D_OT_OpenManual,
    BDENTAL_4D_OT_Template,
    BDENTAL_4D_OT_Organize,
    BDENTAL_4D_OT_Volume_Render,
    BDENTAL_4D_OT_ResetCtVolumePosition,
    BDENTAL_4D_OT_AddSlices,
    BDENTAL_4D_OT_MultiTreshSegment,
    # BDENTAL_4D_OT_MultiView,
    BDENTAL_4D_OT_MPR,
    BDENTAL_4D_OT_AddReferencePlanes,
    BDENTAL_4D_OT_CtVolumeOrientation,
    BDENTAL_4D_OT_AddMarkupPoint,
    BDENTAL_4D_OT_AddTeeth,
    BDENTAL_4D_OT_AddSleeve,
    BDENTAL_4D_OT_AddImplant,
    BDENTAL_4D_OT_AlignImplants,
    BDENTAL_4D_OT_AddImplantSleeve,
    BDENTAL_4D_OT_AddSplint,
    BDENTAL_4D_OT_Survey,
    BDENTAL_4D_OT_BlockModel,
    BDENTAL_4D_OT_ModelBase,
    BDENTAL_4D_OT_add_offset,
    BDENTAL_4D_OT_hollow_model,
    BDENTAL_4D_OT_AlignPoints,
    BDENTAL_4D_OT_AlignPointsInfo,
    BDENTAL_4D_OT_AddColor,
    BDENTAL_4D_OT_RemoveColor,
    BDENTAL_4D_OT_JoinObjects,
    BDENTAL_4D_OT_SeparateObjects,
    BDENTAL_4D_OT_Parent,
    BDENTAL_4D_OT_Unparent,
    BDENTAL_4D_OT_align_to_front,
    BDENTAL_4D_OT_to_center,
    BDENTAL_4D_OT_center_cursor,
    BDENTAL_4D_OT_OcclusalPlane,
    BDENTAL_4D_OT_OcclusalPlaneInfo,
    BDENTAL_4D_OT_decimate,
    BDENTAL_4D_OT_clean_mesh,
    BDENTAL_4D_OT_clean_mesh2,
    BDENTAL_4D_OT_fill,
    BDENTAL_4D_OT_retopo_smooth,
    BDENTAL_4D_OT_VoxelRemesh,
    BDENTAL_4D_OT_CurveCutterAdd,
    BDENTAL_4D_OT_CurveCutterAdd2,
    BDENTAL_4D_OT_CurveCutterAdd3,
    BDENTAL_4D_OT_CurveCutterCut,
    BDENTAL_4D_OT_CurveCutterCut3,
    BDENTAL_4D_OT_CurveCutter2_ShortPath,
    BDENTAL_4D_OT_square_cut,
    BDENTAL_4D_OT_square_cut_confirm,
    BDENTAL_4D_OT_square_cut_exit,
    BDENTAL_4D_OT_SplintCutterAdd,
    BDENTAL_4D_OT_SplintCutterCut,
    BDENTAL_4D_OT_PaintArea,
    BDENTAL_4D_OT_PaintAreaPlus,
    BDENTAL_4D_OT_PaintAreaMinus,
    BDENTAL_4D_OT_PaintCut,
    BDENTAL_4D_OT_AddTube,
    BDENTAL_4D_OT_Matching2D3D,
    BDENTAL_4D_OT_XrayToggle,
]


def register():

    for cls in classes:
        bpy.utils.register_class(cls)
    post_handlers = bpy.app.handlers.depsgraph_update_post
    MyPostHandlers = [
        "BDENTAL4D_TresholdMinUpdate",
        "BDENTAL4D_TresholdMaxUpdate",
        "BDENTAL4D_AxialSliceUpdate",
        "BDENTAL4D_CoronalSliceUpdate",
        "BDENTAL4D_SagitalSliceUpdate",
    ]

    # Remove old handlers :
    handlers_To_Remove = [h for h in post_handlers if h.__name__ in MyPostHandlers]
    if handlers_To_Remove:
        for h in handlers_To_Remove:
            bpy.app.handlers.depsgraph_update_post.remove(h)

    handlers_To_Add = [
        BDENTAL4D_TresholdMinUpdate,
        BDENTAL4D_TresholdMaxUpdate,
        BDENTAL4D_AxialSliceUpdate,
        BDENTAL4D_CoronalSliceUpdate,
        BDENTAL4D_SagitalSliceUpdate,
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
        "BDENTAL4D_TresholdMinUpdate",
        "BDENTAL4D_TresholdMaxUpdate",
        "BDENTAL4D_AxialSliceUpdate",
        "BDENTAL4D_CoronalSliceUpdate",
        "BDENTAL4D_SagitalSliceUpdate",
    ]
    handlers_To_Remove = [h for h in post_handlers if h.__name__ in MyPostHandlers]

    if handlers_To_Remove:
        for h in handlers_To_Remove:
            bpy.app.handlers.depsgraph_update_post.remove(h)

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
