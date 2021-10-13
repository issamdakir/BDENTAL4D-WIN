# Python imports :
import os, sys, shutil, threading, tempfile
from os.path import join, dirname, exists, abspath
import math
from math import degrees, radians, pi, ceil, floor, acos
import numpy as np
from numpy.linalg import svd
from time import sleep, perf_counter as Tcounter
from queue import Queue
from importlib import reload
from bpy.app.handlers import persistent


import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from matplotlib.patches import Arc
from matplotlib.transforms import IdentityTransform, TransformedBbox, Bbox

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


# Blender Imports :
import bpy
import bmesh
import mathutils
from mathutils import Matrix, Vector, Euler, kdtree, geometry as Geo
import gpu
from gpu_extras.batch import batch_for_shader
import bgl
import blf

import SimpleITK as sitk
import vtk
import cv2

try:
    cv2 = reload(cv2)
except ImportError:
    pass
from vtk.util import numpy_support
from vtk import vtkCommand

# Global Variables :
ProgEvent = vtkCommand.ProgressEvent

######################################################################
def GetAutoReconstructParameters(Manufacturer, ConvKernel):
    
    Soft,Bone,Teeth = None, None, None

    if ConvKernel != None :

        if Manufacturer == "NewTom":
            Soft,Bone,Teeth = -400, 606, 1032

        if Manufacturer == "J.Morita.Mfg.Corp."  and ConvKernel == "FBP" :
            Soft,Bone,Teeth = -365, 200,455

        else:
            if ('Hr40f' in ConvKernel and '3' in ConvKernel) or \
            ('J30s' in ConvKernel and '3' in ConvKernel) or \
            ('J30f' in ConvKernel and '2'in ConvKernel) or \
            ('I31f' in ConvKernel and '3' in ConvKernel) or \
            ('Br40f' in ConvKernel and '3' in ConvKernel) or \
            ('Hr38h' in ConvKernel and '3' in ConvKernel) or \
            ConvKernel in ("FC03","FC04","STANDARD","H30s","SOFT","UB", "SA" , "FC23" , "FC08" ,"FC21" , "A" ,"FC02" ,"B" ,"H23s" ,"H20s","H31s" , "H32s"  , "H40s","H31s" , "B41s" , "B70s" , "H22s","H20f" , "FC68" , "FC07" ,"B30s" , "B41s" , "D10f" , "B45s" ,"B26f" , "B30f" , "32" , "SB" ,"FC15" , "FC69" , "UA", "10" ,"STND" , "H30f" , "B20s"):
        
                Soft,Bone,Teeth = -300, 200, 1430

            if ('Hr60f' in ConvKernel and '3' in ConvKernel) or \
            ('I70f' in ConvKernel and '3' in ConvKernel) or \
            ('Hr64h' in ConvKernel and '3' in ConvKernel) or \
            ConvKernel in ("BONE", "BONEPLUS", "FC30", "H70s","D", "EA", "FC81", "YC","YD", "H70h", "H60s", "H60f", "FC35","B80s", "H90s", "B70f", "EB", "11H", "C", "B60s"):

                Soft,Bone,Teeth = -300, 400, 995

    if not ConvKernel :
        
        if Manufacturer == "Imaging Sciences International":
            Soft,Bone,Teeth = -400, 358, 995

        if Manufacturer == "SOREDEX":
            Soft,Bone,Teeth = -400, 410 , 880


        if Manufacturer == "Xoran Technologies ®":
            Soft,Bone,Teeth = -400, 331, 1052

        if Manufacturer == "Planmeca":
            Soft,Bone,Teeth = -400, 330, 756

        if Manufacturer == "J.Morita.Mfg.Corp.":
            Soft,Bone,Teeth = -315, 487,787

        if Manufacturer in ["Carestream Health", 'Carestream Dental']:
            Soft,Bone,Teeth = -400, 388, 1013

        if Manufacturer == "MyRay":
            Soft,Bone,Teeth = -360, 850, 1735

        if Manufacturer == "NIM":
            Soft,Bone,Teeth = -1, 1300, 1260

        if Manufacturer == "PreXion":
            Soft,Bone,Teeth = -400, 312, 1505

        if Manufacturer == "Sirona":
            Soft,Bone,Teeth = -170, 590, 780

        if Manufacturer == "Dabi Atlante":
            Soft,Bone,Teeth = -375, 575, 1080

        if Manufacturer == "INSTRUMENTARIUM DENTAL":
            Soft,Bone,Teeth = -400, 430, 995

        if Manufacturer == "Instrumentarium Dental":
            Soft,Bone,Teeth = -357, 855, 1489

        if Manufacturer == "Vatech Company Limited":
            Soft,Bone,Teeth = -328, 780, 1520

    return Soft,Bone,Teeth




#######################################################################################
# Popup message box function :
#######################################################################################


def ShowMessageBox(message=[], title="INFO", icon="INFO"):
    def draw(self, context):
        for txtLine in message:
            self.layout.label(text=txtLine)

    bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)


#######################################################################################
# Load CT Scan functions :
#######################################################################################
def Align_Implants(Averrage=False):
    ctx = bpy.context
    Implants = ctx.selected_objects
    n = len(Implants)
    Active_Imp = ctx.active_object
    if n <2 or not Active_Imp :
        print('Please select at least 2 implants \nThe last selected is the active')
        return
    else:
        if Averrage :
            MeanRot = np.mean([ np.array(Impt.rotation_euler) for Impt in Implants], axis=0)
            for Impt in Implants :
                Impt.rotation_euler = MeanRot
            
        else :
            
            for Impt in Implants :
                Impt.rotation_euler = Active_Imp.rotation_euler

                
def CheckString(String, MatchesList,mode=all):
    if mode(x in String for x in MatchesList ):
        return True
    else:
        return False
def rmtree(top):
    for root, dirs, files in os.walk(top, topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            os.chmod(filename, stat.S_IWUSR)
            os.remove(filename)
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(top)


def get_all_addons(display=False):
    """
    Prints the addon state based on the user preferences.

    """
    import bpy as _bpy
    from addon_utils import check, paths, enable
    import sys

    # RELEASE SCRIPTS: official scripts distributed in Blender releases
    paths_list = paths()
    addon_list = []
    for path in paths_list:
        _bpy.utils._sys_path_ensure(path)
        for mod_name, mod_path in _bpy.path.module_names(path):
            is_enabled, is_loaded = check(mod_name)
            addon_list.append(mod_name)
            if display:  # for example
                print("%s default:%s loaded:%s" % (mod_name, is_enabled, is_loaded))

    return addon_list


def Addon_Enable(AddonName, Enable=True):
    import addon_utils as AU

    is_enabled, is_loaded = AU.check(AddonName)
    # for mod in AU.modules() :
    # Name = mod.bl_info["name"]
    # print(Name)
    if Enable:
        if not is_enabled:
            AU.enable(AddonName, default_set=True)
    if not Enable:
        if is_enabled:
            AU.disable(AddonName, default_set=True)

    is_enabled, is_loaded = AU.check(AddonName)
    # print(f"{AddonName} : (is_enabled : {is_enabled} , is_loaded : {is_loaded})")


def CleanScanData(Preffix):
    D = bpy.data
    Objects = D.objects
    Meshes = D.meshes
    Images = D.images
    Materials = D.materials
    NodeGroups = D.node_groups

    # Remove Voxel data :
    [Meshes.remove(m) for m in Meshes if f"{Preffix}_PLANE_" in m.name]
    [Images.remove(img) for img in Images if f"{Preffix}_img" in img.name]
    [Materials.remove(mat) for mat in Materials if "BD001_Voxelmat_" in mat.name]
    [NodeGroups.remove(NG) for NG in NodeGroups if "BD001_VGS_" in NG.name]

    # Remove old Slices :
    SlicePlanes = [
        Objects.remove(obj)
        for obj in Objects
        if Preffix in obj.name and "SLICE" in obj.name
    ]
    SliceMeshes = [
        Meshes.remove(m) for m in Meshes if Preffix in m.name and "SLICE" in m.name
    ]
    SliceMats = [
        Materials.remove(mat)
        for mat in Materials
        if Preffix in mat.name and "SLICE" in mat.name
    ]
    SliceImages = [
        Images.remove(img)
        for img in Images
        if Preffix in img.name and "SLICE" in img.name
    ]


def CtxOverride(context):
    area3D = [area for area in context.screen.areas if area.type == "VIEW_3D"][0]
    space3D = [space for space in area3D.spaces if space.type == "VIEW_3D"][0]
    region3D = [reg for reg in area3D.regions if reg.type == "WINDOW"][0]

    Override = {"area": area3D, "space_data": space3D, "region": region3D}
    return Override, area3D, space3D


def AbsPath(P):
    if P.startswith("//"):
        P = abspath(bpy.path.abspath(P))
    return P


def RelPath(P):
    if not P.startswith("//"):
        P = bpy.path.relpath(abspath(P))
    return P


############################
# Make directory function :
############################
def make_directory(Root, DirName):

    DirPath = join(Root, DirName)
    if not DirName in os.listdir(Root):
        os.mkdir(DirPath)
    return DirPath


################################
# Copy DcmSerie To ProjDir function :
################################
def CopyDcmSerieToProjDir(DcmSerie, DicomSeqDir):
    for i in range(len(DcmSerie)):
        shutil.copy2(DcmSerie[i], DicomSeqDir)


##########################################################################################
######################### BDENTAL_4D Volume Render : ########################################
##########################################################################################
def AddMaterial(Obj, matName, color, transparacy=None):

    mat = bpy.data.materials.get(matName) or bpy.data.materials.new(matName)
    mat.use_nodes = False
    mat.diffuse_color = color
    Obj.active_material = mat

    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = color
    mat.blend_method = "BLEND"

    if transparacy:
        mat.node_tree.nodes["Principled BSDF"].inputs[19].default_value = 0.5


def PlaneCut(Target, Plane, inner=False, outer=False, fill=False):

    bpy.ops.object.select_all(action="DESELECT")
    Target.select_set(True)
    bpy.context.view_layer.objects.active = Target

    Pco = Plane.matrix_world.translation
    Pno = Plane.matrix_world.to_3x3().transposed()[2]

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.bisect(
        plane_co=Pco,
        plane_no=Pno,
        use_fill=fill,
        clear_inner=inner,
        clear_outer=outer,
    )
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")


def AddBooleanCube(DimX, DimY, DimZ):
    bpy.ops.mesh.primitive_cube_add(
        size=max(DimX, DimY, DimZ) * 1.5,
        enter_editmode=False,
        align="WORLD",
        location=(0, 0, 0),
        scale=(1, 1, 1),
    )

    VOI = VolumeOfInterst = bpy.context.object
    VOI.name = "VOI"
    VOI.display_type = "WIRE"
    return VOI


def AddNode(nodes, type, name):

    node = nodes.new(type)
    node.name = name
    # node.location[0] -= 200

    return node


def AddFrankfortPoint(PointsList, color, CollName):
    FrankfortPointsNames = ["R_Or", "L_Or", "R_Po", "L_Po"]
    Loc = bpy.context.scene.cursor.location
    if not PointsList:
        P = AddMarkupPoint(FrankfortPointsNames[0], color, Loc, 1, CollName)
        return P
    if PointsList:
        CurrentPointsNames = [P.name for P in PointsList]
        P_Names = [P for P in FrankfortPointsNames if not P in CurrentPointsNames]
        if P_Names:
            P = AddMarkupPoint(P_Names[0], color, Loc, 1, CollName)
            return P
    else:
        return None


def AddMarkupPoint(name, color, loc, Diameter=1, CollName=None, show_name=False):

    bpy.ops.mesh.primitive_uv_sphere_add(radius=Diameter / 2, location=loc)
    P = bpy.context.object
    P.name = name
    P.data.name = name + "_mesh"

    if CollName:
        MoveToCollection(P, CollName)

    matName = f"{name}_Mat"
    mat = bpy.data.materials.get(matName) or bpy.data.materials.new(matName)
    mat.diffuse_color = color
    mat.use_nodes = False
    P.active_material = mat
    P.show_name = show_name
    return P


def ProjectPoint(Plane, Point):

    V1, V2, V3, V4 = [Plane.matrix_world @ V.co for V in Plane.data.vertices]
    Ray = Plane.matrix_world.to_3x3() @ Plane.data.polygons[0].normal

    Orig = Point
    Result = Geo.intersect_ray_tri(V1, V2, V3, Ray, Orig, False)
    if not Result:
        Ray *= -1
        Result = Geo.intersect_ray_tri(V1, V2, V3, Ray, Orig, False)
    return Result


# def PointsToFrankfortPlane(ctx, Model, CurrentPointsList, color, CollName=None):

#     Dim = max(Model.dimensions) * 1.5
#     Na, R_Or, L_Or, R_Po, L_Po = [P.location for P in CurrentPointsList]

#     Center = (R_Or + L_Or + R_Po + L_Po) / 4

#     FrankZ = (R_Or - Center).cross((L_Or - Center)).normalized()
#     FrankX = (Center - Na).cross(FrankZ).normalized()
#     FrankY = FrankZ.cross(FrankX).normalized()

#     FrankMtx = Matrix((FrankX, FrankY, FrankZ)).to_4x4().transposed()
#     FrankMtx.translation = Center

#     SagZ = -FrankX
#     SagX = FrankZ
#     SagY = FrankY

#     SagMtx = Matrix((SagX, SagY, SagZ)).to_4x4().transposed()
#     SagMtx.translation = Center

#     CorZ = -FrankY
#     CorX = FrankX
#     CorY = FrankZ

#     CorMtx = Matrix((CorX, CorY, CorZ)).to_4x4().transposed()
#     CorMtx.translation = Center

#     bpy.ops.mesh.primitive_plane_add(ctx, size=Dim)
#     FrankfortPlane = bpy.context.object
#     name = "Frankfort_Plane"
#     FrankfortPlane.name = name
#     FrankfortPlane.data.name = f"{name}_Mesh"
#     FrankfortPlane.matrix_world = FrankMtx
#     matName = f"RefPlane_Mat"
#     mat = bpy.data.materials.get(matName) or bpy.data.materials.new(matName)
#     mat.diffuse_color = color
#     mat.use_nodes = False
#     FrankfortPlane.active_material = mat

#     if CollName:
#         MoveToCollection(FrankfortPlane, CollName)
#     # Add Sagital Plane :
#     bpy.ops.object.duplicate_move()
#     SagPlane = bpy.context.object
#     name = "Sagital_Median_Plane"
#     SagPlane.name = name
#     SagPlane.data.name = f"{name}_Mesh"
#     SagPlane.matrix_world = SagMtx
#     if CollName:
#         MoveToCollection(SagPlane, CollName)

#     # Add Coronal Plane :
#     bpy.ops.object.duplicate_move()
#     CorPlane = bpy.context.object
#     name = "Coronal_Plane"
#     CorPlane.name = name
#     CorPlane.data.name = f"{name}_Mesh"
#     CorPlane.matrix_world = CorMtx
#     if CollName:
#         MoveToCollection(CorPlane, CollName)

#     # Project Na to Coronal Plane :
#     Na_Projection_1 = ProjectPoint(Plane=CorPlane, Point=Na)

#     # Project Na_Projection_1 to frankfort Plane :
#     Na_Projection_2 = ProjectPoint(Plane=FrankfortPlane, Point=Na_Projection_1)

#     for Plane in (FrankfortPlane, CorPlane, SagPlane):
#         Plane.location = Na_Projection_2
#     return [FrankfortPlane, SagPlane, CorPlane]


def PointsToPlaneMatrix(Ant_Point, PointsList):
    points = np.array(PointsList)
    if not points.shape[0] >= points.shape[1]:
        print("points Array should be of shape (n,m) :")
        print("where n is the number of points and m the dimension( x,y,z = 3)  ")
        return
    C = points.mean(axis=0)
    x = points - C
    M = np.dot(x.T, x)  # Could also use np.cov(x) here.
    N = svd(M)[0][:, -1]

    Center, Normal = Vector(C), Vector(N)

    PlaneZ = Normal
    PlaneX = ((Center - Ant_Point).cross(PlaneZ)).normalized()
    PlaneY = (PlaneZ.cross(PlaneX)).normalized()

    return PlaneX, PlaneY, PlaneZ, Center


def PointsToRefPlanes(ctx, Model, RefPointsList, color, CollName=None):
    Dim = max(Model.dimensions) * 1.5
    Na, R_Or, L_Or, R_Po, L_Po = [P.location for P in RefPointsList]
    PlaneX, PlaneY, PlaneZ, Center = PointsToPlaneMatrix(
        Ant_Point=Na, PointsList=[R_Or, L_Or, R_Po, L_Po]
    )
    # Frankfort Ref Plane :
    FrankX = PlaneX
    FrankY = PlaneY
    FrankZ = PlaneZ

    FrankMtx = Matrix((FrankX, FrankY, FrankZ)).to_4x4().transposed()
    FrankMtx.translation = Center

    # Sagital Median Plane :
    SagZ = -FrankX
    SagX = FrankZ
    SagY = FrankY

    SagMtx = Matrix((SagX, SagY, SagZ)).to_4x4().transposed()
    SagMtx.translation = Center

    # Coronal(Frontal) Plane :
    CorZ = -FrankY
    CorX = FrankX
    CorY = FrankZ

    CorMtx = Matrix((CorX, CorY, CorZ)).to_4x4().transposed()
    CorMtx.translation = Center

    # Add Planes :
    bpy.ops.mesh.primitive_plane_add(ctx, size=Dim)
    FrankfortPlane = bpy.context.object
    name = "01-Frankfort_Plane"
    FrankfortPlane.name = name
    FrankfortPlane.data.name = f"{name}_Mesh"
    FrankfortPlane.matrix_world = FrankMtx
    matName = f"BDENTAL_4D_RefPlane_Mat"
    mat = bpy.data.materials.get(matName) or bpy.data.materials.new(matName)
    mat.use_nodes = False
    mat.diffuse_color = color
    FrankfortPlane.active_material = mat

    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = color
    mat.node_tree.nodes["Principled BSDF"].inputs[19].default_value = 0.5
    mat.blend_method = "BLEND"

    if CollName:
        MoveToCollection(FrankfortPlane, CollName)
    # Add Sagital Plane :
    bpy.ops.object.duplicate_move()
    SagPlane = bpy.context.object
    name = "02-Sagital_Median_Plane"
    SagPlane.name = name
    SagPlane.data.name = f"{name}_Mesh"
    SagPlane.matrix_world = SagMtx
    if CollName:
        MoveToCollection(SagPlane, CollName)

    # Add Coronal Plane :
    bpy.ops.object.duplicate_move()
    CorPlane = bpy.context.object
    name = "03-Coronal_Plane"
    CorPlane.name = name
    CorPlane.data.name = f"{name}_Mesh"
    CorPlane.matrix_world = CorMtx
    if CollName:
        MoveToCollection(CorPlane, CollName)

    # Project Na to Coronal Plane :
    Na_Projection_1 = ProjectPoint(Plane=CorPlane, Point=Na)

    # Project Na_Projection_1 to frankfort Plane :
    Na_Projection_2 = ProjectPoint(Plane=FrankfortPlane, Point=Na_Projection_1)

    for Plane in (FrankfortPlane, CorPlane, SagPlane):
        Plane.location = Na_Projection_2

    return [FrankfortPlane, SagPlane, CorPlane]


def PointsToOcclusalPlane(ctx, Model, R_pt, A_pt, L_pt, color, subdiv):

    Dim = max(Model.dimensions) * 1.2

    Rco = R_pt.location
    Aco = A_pt.location
    Lco = L_pt.location

    Center = (Rco + Aco + Lco) / 3

    Z = (Rco - Center).cross((Aco - Center)).normalized()
    X = Z.cross((Aco - Center)).normalized()
    Y = Z.cross(X).normalized()

    Mtx = Matrix((X, Y, Z)).to_4x4().transposed()
    Mtx.translation = Center

    bpy.ops.mesh.primitive_plane_add(ctx, size=Dim)
    OcclusalPlane = bpy.context.object
    name = "Occlusal_Plane"
    OcclusalPlane.name = name
    OcclusalPlane.data.name = f"{name}_Mesh"
    OcclusalPlane.matrix_world = Mtx

    matName = f"{name}_Mat"
    mat = bpy.data.materials.get(matName) or bpy.data.materials.new(matName)
    mat.diffuse_color = color
    mat.use_nodes = False
    OcclusalPlane.active_material = mat
    if subdiv:
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.subdivide(number_cuts=50)
        bpy.ops.object.mode_set(mode="OBJECT")

    return OcclusalPlane


##############################################
def TriPlanes_Point_Intersect(P1, P2, P3, CrossLenght):

    P1N = P1.matrix_world.to_3x3() @ P1.data.polygons[0].normal
    P2N = P2.matrix_world.to_3x3() @ P2.data.polygons[0].normal
    P3N = P3.matrix_world.to_3x3() @ P3.data.polygons[0].normal

    Condition = np.dot(np.array(P1N), np.cross(np.array(P2N), np.array(P3N))) != 0

    C1, C2, C3 = P1.location, P2.location, P3.location

    F1 = sum(list(P1.location * P1N))
    F2 = sum(list(P2.location * P2N))
    F3 = sum(list(P3.location * P3N))

    # print(Matrix((P1N,P2N,P3N)))
    if Condition:

        P_Intersect = Matrix((P1N, P2N, P3N)).inverted() @ Vector((F1, F2, F3))
        P1P2_Vec = (
            Vector(np.cross(np.array(P1N), np.array(P2N))).normalized() * CrossLenght
        )
        P2P3_Vec = (
            Vector(np.cross(np.array(P2N), np.array(P3N))).normalized() * CrossLenght
        )
        P1P3_Vec = (
            Vector(np.cross(np.array(P1N), np.array(P3N))).normalized() * CrossLenght
        )

        P1P2 = [P_Intersect + P1P2_Vec, P_Intersect - P1P2_Vec]
        P2P3 = [P_Intersect + P2P3_Vec, P_Intersect - P2P3_Vec]
        P1P3 = [P_Intersect + P1P3_Vec, P_Intersect - P1P3_Vec]

        return P_Intersect, P1P2, P2P3, P1P3
    else:
        return None


###########################################################
def AddPlaneMesh(DimX, DimY, Name):
    x = DimX / 2
    y = DimY / 2
    verts = [(-x, -y, 0.0), (x, -y, 0.0), (-x, y, 0.0), (x, y, 0.0)]
    faces = [(0, 1, 3, 2)]
    mesh_data = bpy.data.meshes.new(f"{Name}_mesh")
    mesh_data.from_pydata(verts, [], faces)
    uvs = mesh_data.uv_layers.new(name=f"{Name}_uv")
    # Returns True if any invalid geometry was removed.
    corrections = mesh_data.validate(verbose=True, clean_customdata=True)
    # Load BMesh with mesh data.
    bm = bmesh.new()
    bm.from_mesh(mesh_data)
    bm.to_mesh(mesh_data)
    bm.free()
    mesh_data.update(calc_edges=True, calc_edges_loose=True)

    return mesh_data


def AddPlaneObject(Name, mesh, CollName):
    Plane_obj = bpy.data.objects.new(Name, mesh)
    MyColl = bpy.data.collections.get(CollName)

    if not MyColl:
        MyColl = bpy.data.collections.new(CollName)

    if not MyColl in bpy.context.scene.collection.children[:]:
        bpy.context.scene.collection.children.link(MyColl)

    if not Plane_obj in MyColl.objects[:]:
        MyColl.objects.link(Plane_obj)

    return Plane_obj


def MoveToCollection(obj, CollName):

    OldColl = obj.users_collection  # list of all collection the obj is in
    NewColl = bpy.data.collections.get(CollName)
    if not NewColl:
        NewColl = bpy.data.collections.new(CollName)
        bpy.context.scene.collection.children.link(NewColl)
    if not obj in NewColl.objects[:]:
        NewColl.objects.link(obj)  # link obj to scene
    if OldColl:
        for Coll in OldColl:  # unlink from all  precedent obj collections
            if Coll is not NewColl:
                Coll.objects.unlink(obj)
    return NewColl

@persistent
def BDENTAL4D_TresholdMinUpdate(scene):

    CtVolumeList = [
        obj
        for obj in bpy.context.scene.objects
        if ("BD4D" in obj.name and "_CTVolume" in obj.name)
    ]
    if CtVolumeList:
        BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props
        GpShader = BDENTAL_4D_Props.GroupNodeName
        Active_Obj = bpy.context.view_layer.objects.active
        if Active_Obj and Active_Obj in CtVolumeList:
            # print('Trshold update trigred')
            Vol = Active_Obj
            Preffix = Vol.name[:10]
            GpNode = bpy.data.node_groups.get(f"{Preffix}_{GpShader}")
            if GpNode :
                Low_Treshold = GpNode.nodes["Low_Treshold"].outputs[0]
                BDENTAL_4D_Props.TresholdMin = Low_Treshold.default_value
            
@persistent
def BDENTAL4D_TresholdMaxUpdate(scene):

    CtVolumeList = [
        obj
        for obj in bpy.context.scene.objects
        if ("BD4D" in obj.name and "_CTVolume" in obj.name)
    ]
    if CtVolumeList:
        BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props
        GpShader = BDENTAL_4D_Props.GroupNodeName
        Active_Obj = bpy.context.view_layer.objects.active
        if Active_Obj and Active_Obj in CtVolumeList:
            # print('Trshold update trigred')
            Vol = Active_Obj
            Preffix = Vol.name[:10]
            GpNode = bpy.data.node_groups.get(f"{Preffix}_{GpShader}")
            if GpNode :
                High_Treshold = GpNode.nodes["High_Treshold"].outputs[0]
                BDENTAL_4D_Props.TresholdMax = High_Treshold.default_value

def BackupArray_To_3DSitkImage(DcmInfo):
    BackupList = eval(DcmInfo['BackupArray'])
    BackupArray = np.array(BackupList)
    Sp, Origin, Direction = DcmInfo["Spacing"], DcmInfo["Origin"],DcmInfo["Direction"]
    img3D = sitk.GetImageFromArray(BackupArray)
    img3D.SetDirection(Direction)
    img3D.SetSpacing(Sp)
    img3D.SetOrigin(Origin)
    return img3D
                
                
def VolumeRender(DcmInfo, GpShader, ShadersBlendFile, VoxelMode):

    Preffix = DcmInfo["Preffix"]

    Sp = Spacing = DcmInfo["RenderSp"]
    Sz = Size = DcmInfo["RenderSz"]
    TransformMatrix = DcmInfo["TransformMatrix"]
    DimX, DimY, DimZ = (Sz[0] * Sp[0], Sz[1] * Sp[1], Sz[2] * Sp[2])
    SagitalOffset, CoronalOffset, AxialOffset= Sp

    AxialPlansList, CoronalPlansList, SagitalPlansList = [],[],[]
    # Load VGS Group Node :
    VGS = bpy.data.node_groups.get(f"{Preffix}_{GpShader}")
    if not VGS:
        filepath = join(ShadersBlendFile, "NodeTree", GpShader)
        directory = join(ShadersBlendFile, "NodeTree")
        filename = GpShader
        bpy.ops.wm.append(filepath=filepath, filename=filename, directory=directory)
        VGS = bpy.data.node_groups.get(GpShader)
        VGS.name = f"{Preffix}_{GpShader}"
        VGS = bpy.data.node_groups.get(f"{Preffix}_{GpShader}")

    Override, area3D, space3D = CtxOverride(bpy.context)
    ###################### Change to ORTHO persp with nice view angle :##########
    
    ViewMatrix = Matrix(
        (
            (0.8677, -0.4971, 0.0000, 8),
            (0.4080, 0.7123, 0.5711, -28),
            (-0.2839, -0.4956, 0.8209, -188),
            (0.0000, 0.0000, 0.0000, 1.0000),
        )
    )
    for scr in bpy.data.screens:
        # if scr.name in ["Layout", "Scripting", "Shading"]:
        for area in [ar for ar in scr.areas if ar.type == "VIEW_3D"]:
            for space in [sp for sp in area.spaces if sp.type == "VIEW_3D"]:
                r3d = space.region_3d
                r3d.view_perspective = "ORTHO"
                # r3d.view_distance = 800
                r3d.view_matrix = ViewMatrix
                space.shading.type = "SOLID"
                space.shading.color_type = "TEXTURE"
                space.overlay.show_overlays = False
                r3d.update()
    ######################################################################################
    #Axial Voxels :
    ######################################################################################

    AxialImagesNamesList = sorted(
        [img.name for img in bpy.data.images if img.name.startswith(f"{Preffix}_Axial_")]
        )
    AxialImagesList = [bpy.data.images[Name] for Name in AxialImagesNamesList]
        
    print("Axial Voxel rendering...")
    for i, ImageData in enumerate(AxialImagesList):
        # # Add Plane :
        # ##########################################
        Name = f"{Preffix}_Axial_PLANE_{i}"
        mesh = AddPlaneMesh(DimX, DimY, Name)
        CollName = "CT_Voxel"

        obj = AddPlaneObject(Name, mesh, CollName)
        obj.location = (0,0,i * AxialOffset)
        AxialPlansList.append(obj)

        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        ##########################################
        # Add Material :
        mat = bpy.data.materials.new(f"{Preffix}_Axial_Voxelmat_{i}")
        mat.use_nodes = True
        node_tree = mat.node_tree
        nodes = node_tree.nodes
        links = node_tree.links

        for node in nodes:
            if node.type != "OUTPUT_MATERIAL":
                nodes.remove(node)

        # ImageData = bpy.data.images.get(ImagePNG)
        TextureCoord = AddNode(nodes, type="ShaderNodeTexCoord", name="TextureCoord")
        ImageTexture = AddNode(nodes, type="ShaderNodeTexImage", name="Image Texture")

        ImageTexture.image = ImageData
        ImageTexture.extension = 'CLIP'

        ImageData.colorspace_settings.name = "Non-Color"

        materialOutput = nodes["Material Output"]

        links.new(TextureCoord.outputs[0], ImageTexture.inputs[0])

        GroupNode = nodes.new("ShaderNodeGroup")
        GroupNode.node_tree = VGS

        links.new(ImageTexture.outputs["Color"], GroupNode.inputs[0])
        links.new(GroupNode.outputs[0], materialOutput.inputs["Surface"])
        for _ in obj.material_slots:
            bpy.ops.object.material_slot_remove()

        obj.active_material = mat

        mat.blend_method = "HASHED"#"BLEND"
        mat.shadow_method = "HASHED"
        # mat.use_backface_culling = True

        Override, area3D, space3D = CtxOverride(bpy.context)
        if i==0 :
            bpy.ops.view3d.view_selected(Override)

        
        # bpy.ops.wm.redraw_timer(type='DRAW_SWAP',iterations=1)
        # bpy.ops.wm.redraw_timer(Override, type='DRAW_WIN_SWAP', iterations=1)

        ############# LOOP END ##############
        #####################################

    # Join Planes Make Cube Voxel :
    bpy.ops.object.select_all(action="DESELECT")
    for obj in AxialPlansList:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
    
    Col = bpy.data.collections.get("CT_Voxel")
    if Col :
        Col.hide_viewport = False
    
    bpy.ops.object.join()

    Voxel_Axial = bpy.context.object

    Voxel_Axial.name = f"{Preffix}_Axial_CTVolume"
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")
    Voxel_Axial.matrix_world.translation = (0,0,0)

    Voxel_Axial.matrix_world = TransformMatrix @ Voxel_Axial.matrix_world 

    for i in range(3):
        Voxel_Axial.lock_location[i] = True
        Voxel_Axial.lock_rotation[i] = True
        Voxel_Axial.lock_scale[i] = True

    # Voxel_Axial.hide_set(True)
    ###############################################################################
    #Coronal Voxels :
    ###############################################################################
    if VoxelMode in ["OPTIMAL", "FULL"] :
        print("Coronal Voxel rendering...")
        CoronalImagesNamesList = sorted(
            [img.name for img in bpy.data.images if img.name.startswith(f"{Preffix}_Coronal_")]
        )
        CoronalImagesList = [bpy.data.images[Name] for Name in CoronalImagesNamesList]

            
        for i, ImageData in enumerate(CoronalImagesList):
            # # Add Plane :
            # ##########################################
            Name = f"{Preffix}_Coronal_PLANE_{i}"
            mesh = AddPlaneMesh(DimX, DimZ, Name)
            CollName = "CT_Voxel"

            obj = AddPlaneObject(Name, mesh, CollName)
            obj.location = (0,0,-i * CoronalOffset)
            CoronalPlansList.append(obj)

            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj

            ##########################################
            # Add Material :
            mat = bpy.data.materials.new(f"{Preffix}_Coronal_Voxelmat_{i}")
            mat.use_nodes = True
            node_tree = mat.node_tree
            nodes = node_tree.nodes
            links = node_tree.links

            for node in nodes:
                if node.type != "OUTPUT_MATERIAL":
                    nodes.remove(node)

            # ImageData = bpy.data.images.get(ImagePNG)
            TextureCoord = AddNode(nodes, type="ShaderNodeTexCoord", name="TextureCoord")
            ImageTexture = AddNode(nodes, type="ShaderNodeTexImage", name="Image Texture")

            ImageTexture.image = ImageData
            ImageData.colorspace_settings.name = "Non-Color"

            materialOutput = nodes["Material Output"]

            links.new(TextureCoord.outputs[0], ImageTexture.inputs[0])

            GroupNode = nodes.new("ShaderNodeGroup")
            GroupNode.node_tree = VGS

            links.new(ImageTexture.outputs["Color"], GroupNode.inputs[0])
            links.new(GroupNode.outputs[0], materialOutput.inputs["Surface"])
            for _ in obj.material_slots:
                bpy.ops.object.material_slot_remove()

            obj.active_material = mat

            mat.blend_method = "HASHED"
            mat.shadow_method = "HASHED"
            # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            ############# LOOP END ##############
            #####################################

        # Join Planes Make Cube Voxel :
        bpy.ops.object.select_all(action="DESELECT")
        for obj in CoronalPlansList:
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
        bpy.context.view_layer.layer_collection.children["CT_Voxel"].hide_viewport = False
        bpy.ops.object.join()

        Voxel_Coronal = bpy.context.object

        Voxel_Coronal.name = f"{Preffix}_Coronal_CTVolume"
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")
        Voxel_Coronal.matrix_world.translation = (0,0,0)
        bpy.ops.transform.rotate(value=1.5708, orient_axis='X', orient_type='GLOBAL')
        

        Voxel_Coronal.matrix_world = TransformMatrix @ Voxel_Coronal.matrix_world
        
        for i in range(3):
            Voxel_Coronal.lock_location[i] = True
            Voxel_Coronal.lock_rotation[i] = True
            Voxel_Coronal.lock_scale[i] = True

        Voxel_Coronal.hide_set(True)

    ###############################################################################
    #Sagital Voxels :
    ###############################################################################
    if VoxelMode == "FULL" :
        print("Sagital Voxel rendering...")
        SagitalImagesNamesList = sorted(
            [img.name for img in bpy.data.images if img.name.startswith(f"{Preffix}_Sagital_")]
        )
        SagitalImagesList = [bpy.data.images[Name] for Name in SagitalImagesNamesList]

        for i, ImageData in enumerate(SagitalImagesList):
            # # Add Plane :
            # ##########################################
            Name = f"{Preffix}_Sagital_PLANE_{i}"
            mesh = AddPlaneMesh(DimY, DimZ, Name)
            CollName = "CT_Voxel"

            obj = AddPlaneObject(Name, mesh, CollName)
            obj.location = (0,0,i * SagitalOffset)
            SagitalPlansList.append(obj)

            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj

            ##########################################
            # Add Material :
            mat = bpy.data.materials.new(f"{Preffix}_Sagital_Voxelmat_{i}")
            mat.use_nodes = True
            node_tree = mat.node_tree
            nodes = node_tree.nodes
            links = node_tree.links

            for node in nodes:
                if node.type != "OUTPUT_MATERIAL":
                    nodes.remove(node)

            # ImageData = bpy.data.images.get(ImagePNG)
            TextureCoord = AddNode(nodes, type="ShaderNodeTexCoord", name="TextureCoord")
            ImageTexture = AddNode(nodes, type="ShaderNodeTexImage", name="Image Texture")

            ImageTexture.image = ImageData
            ImageData.colorspace_settings.name = "Non-Color"

            materialOutput = nodes["Material Output"]

            links.new(TextureCoord.outputs[0], ImageTexture.inputs[0])

            GroupNode = nodes.new("ShaderNodeGroup")
            GroupNode.node_tree = VGS

            links.new(ImageTexture.outputs["Color"], GroupNode.inputs[0])
            links.new(GroupNode.outputs[0], materialOutput.inputs["Surface"])
            for _ in obj.material_slots:
                bpy.ops.object.material_slot_remove()

            obj.active_material = mat

            mat.blend_method = "HASHED"
            mat.shadow_method = "HASHED"
            # bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            ############# LOOP END ##############
            #####################################

        # Join Planes Make Cube Voxel :
        bpy.ops.object.select_all(action="DESELECT")
        for obj in SagitalPlansList:
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
        bpy.context.view_layer.layer_collection.children["CT_Voxel"].hide_viewport = False
        bpy.ops.object.join()

        Voxel_Sagital = bpy.context.object

        Voxel_Sagital.name = f"{Preffix}_Sagital_CTVolume"
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")
        Voxel_Sagital.matrix_world.translation = (0,0,0)

        bpy.ops.transform.rotate(value=1.5708, orient_axis='X', orient_type='GLOBAL')
        bpy.ops.transform.rotate(value=1.5708, orient_axis='Z', orient_type='GLOBAL')
        

        Voxel_Sagital.matrix_world = TransformMatrix @ Voxel_Sagital.matrix_world
        
        for i in range(3):
            Voxel_Sagital.lock_location[i] = True
            Voxel_Sagital.lock_rotation[i] = True
            Voxel_Sagital.lock_scale[i] = True

        Voxel_Sagital.hide_set(True)

    Voxel_Axial.hide_set(False)
    ######################## Set Render settings : #############################
    Scene_Settings()
    ###############################################################################################
    bpy.ops.object.select_all(action="DESELECT")
    Voxel_Axial.select_set(True)
    bpy.context.view_layer.objects.active = Voxel_Axial

    # ###################### Change to ORTHO persp with nice view angle :##########
    
    # ViewMatrix = Matrix(
    #     (
    #         (0.8677, -0.4971, 0.0000, 4.0023),
    #         (0.4080, 0.7123, 0.5711, -14.1835),
    #         (-0.2839, -0.4956, 0.8209, -94.0148),
    #         (0.0000, 0.0000, 0.0000, 1.0000),
    #     )
    # )
    for scr in bpy.data.screens:
        # if scr.name in ["Layout", "Scripting", "Shading"]:
        for area in [ar for ar in scr.areas if ar.type == "VIEW_3D"]:
            for space in [sp for sp in area.spaces if sp.type == "VIEW_3D"]:
                r3d = space.region_3d
                space.shading.type = "MATERIAL"
                space.overlay.show_overlays = True

                r3d.update()

    # Override, area3D, space3D = CtxOverride(bpy.context)
    # bpy.ops.view3d.view_selected(Override)
    Override, area3D, space3D = CtxOverride(bpy.context)
    bpy.ops.view3d.view_selected(Override)
    bpy.ops.object.select_all(action="DESELECT")


def Scene_Settings():
    # Set World Shader node :
    WorldNodes = bpy.data.worlds["World"].node_tree.nodes
    WColor = WorldNodes["Background"].inputs[0].default_value = (0.6, 0.6, 0.6, 0.6)
    WStrength = WorldNodes["Background"].inputs[1].default_value = 1.5

    Override, area3D, space3D = CtxOverride(bpy.context)
    # scene shading lights

    # 3DView Shading Methode : in {'WIREFRAME', 'SOLID', 'MATERIAL', 'RENDERED'}
    space3D.shading.type = "MATERIAL"

    # 'Material' Shading Light method :
    space3D.shading.use_scene_lights = True
    space3D.shading.use_scene_world = False

    # 'RENDERED' Shading Light method :
    space3D.shading.use_scene_lights_render = False
    space3D.shading.use_scene_world_render = True

    space3D.shading.studio_light = "forest.exr"
    space3D.shading.studiolight_rotate_z = 0
    space3D.shading.studiolight_intensity = 1.2
    space3D.shading.studiolight_background_alpha = 0.0
    space3D.shading.studiolight_background_blur = 0.0

    space3D.shading.render_pass = "COMBINED"

    space3D.shading.type = "SOLID"

    # Override, area3D, space3D = CtxOverride(bpy.context)
    space3D.shading.color_type = "TEXTURE"
    # space.shading.light = "MATCAP"
    # space.shading.studio_light = "basic_side.exr"
    space3D.shading.light = "STUDIO"
    space3D.shading.studio_light = "outdoor.sl"
    space3D.shading.show_cavity = True
    space3D.shading.curvature_ridge_factor = 0.5
    space3D.shading.curvature_valley_factor = 0.5

    scn = bpy.context.scene
    scn.render.engine = "BLENDER_EEVEE"
    scn.eevee.use_gtao = True
    scn.eevee.gtao_distance = 12
    scn.eevee.gtao_factor = 1.2
    scn.eevee.gtao_quality = 0.0
    scn.eevee.use_gtao_bounce = False
    scn.eevee.use_gtao_bent_normals = False
    scn.eevee.shadow_cube_size = "512"
    scn.eevee.shadow_cascade_size = "512"
    scn.eevee.use_soft_shadows = True
    scn.eevee.taa_samples = 16
    scn.display_settings.display_device = "None"
    scn.view_settings.look = "Medium Contrast"
    scn.view_settings.exposure = 0.0
    scn.view_settings.gamma = 1.0
    scn.eevee.use_ssr = True


#################################################################################################
# Add Slices :
#################################################################################################
####################################################################
def AddAxialSlice(Preffix, DcmInfo, SlicesDir):
    CTVolume = bpy.context.active_object
    name = f"1_{Preffix}_AXIAL_SLICE"
    Sp, Sz, Origin, Direction, VC = (
        DcmInfo["Spacing"],
        DcmInfo["Size"],
        DcmInfo["Origin"],
        DcmInfo["Direction"],
        DcmInfo["VolumeCenter"],
    )

    DimX, DimY, DimZ = (Sz[0] * Sp[0], Sz[1] * Sp[1], Sz[2] * Sp[2])

    # Remove old Slices and their data meshs :
    OldSlices = [obj for obj in bpy.context.view_layer.objects if name in obj.name]
    OldSliceMeshs = [mesh for mesh in bpy.data.meshes if name in mesh.name]

    for obj in OldSlices:
        bpy.data.objects.remove(obj)
    for mesh in OldSliceMeshs:
        bpy.data.meshes.remove(mesh)

    # Add AXIAL :
    bpy.ops.mesh.primitive_plane_add()
    AxialPlane = bpy.context.active_object
    AxialPlane.name = name
    AxialPlane.data.name = f"{name}_mesh"
    AxialPlane.rotation_mode = "XYZ"
    AxialDims = Vector((DimX, DimY, 0.0))
    AxialPlane.dimensions = AxialDims
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    # AxialPlane.location = VC
    AxialPlane.matrix_world = CTVolume.matrix_world
    # Add Material :
    mat = bpy.data.materials.get(f"{name}_mat") or bpy.data.materials.new(f"{name}_mat")

    for slot in AxialPlane.material_slots:
        bpy.ops.object.material_slot_remove()
    bpy.ops.object.material_slot_add()
    AxialPlane.active_material = mat

    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    for node in nodes:
        if node.type != "OUTPUT_MATERIAL":
            nodes.remove(node)
    ImageName = f"{name}.png"
    ImagePath = join(SlicesDir, ImageName)

    # write "1_AXIAL_SLICE.png" to here ImagePath
    BDENTAL4D_AxialSliceUpdate(bpy.context.scene)

    BlenderImage = bpy.data.images.get(ImageName) or bpy.data.images.load(ImagePath)

    TextureCoord = AddNode(nodes, type="ShaderNodeTexCoord", name="TextureCoord")
    ImageTexture = AddNode(nodes, type="ShaderNodeTexImage", name="Image Texture")
    print(ImageTexture)
    ImageTexture.image = BlenderImage
    BlenderImage.colorspace_settings.name = "Non-Color"
    materialOutput = nodes["Material Output"]
    links.new(TextureCoord.outputs[0], ImageTexture.inputs[0])
    links.new(ImageTexture.outputs[0], materialOutput.inputs[0])
    bpy.context.scene.transform_orientation_slots[0].type = "LOCAL"
    bpy.context.scene.transform_orientation_slots[1].type = "LOCAL"
    bpy.ops.wm.tool_set_by_id(name="builtin.move")

    return AxialPlane


def AddCoronalSlice(Preffix, DcmInfo, SlicesDir):
    
    CTVolume = bpy.context.active_object
    name = f"2_{Preffix}_CORONAL_SLICE"
    Sp, Sz, Origin, Direction, VC = (
        DcmInfo["Spacing"],
        DcmInfo["Size"],
        DcmInfo["Origin"],
        DcmInfo["Direction"],
        DcmInfo["VolumeCenter"],
    )

    DimX, DimY, DimZ = (Sz[0] * Sp[0], Sz[1] * Sp[1], Sz[2] * Sp[2])

    # Remove old Slices and their data meshs :
    OldSlices = [obj for obj in bpy.context.view_layer.objects if name in obj.name]
    OldSliceMeshs = [mesh for mesh in bpy.data.meshes if name in mesh.name]

    for obj in OldSlices:
        bpy.data.objects.remove(obj)
    for mesh in OldSliceMeshs:
        bpy.data.meshes.remove(mesh)

    # Add CORONAL :
    bpy.ops.mesh.primitive_plane_add()
    CoronalPlane = bpy.context.active_object
    CoronalPlane.name = name
    CoronalPlane.data.name = f"{name}_mesh"
    CoronalPlane.rotation_mode = "XYZ"
    CoronalDims = Vector((DimX, DimY, 0.0))
    CoronalPlane.dimensions = CoronalDims
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    rotation_euler = Euler((pi / 2, 0.0, 0.0), "XYZ")
    RotMtx = rotation_euler.to_matrix().to_4x4()
    CoronalPlane.matrix_world = CTVolume.matrix_world @ RotMtx
    # CoronalPlane.rotation_euler = Euler((pi / 2, 0.0, 0.0), "XYZ")
    # CoronalPlane.location = VC
    # Add Material :
    mat = bpy.data.materials.get(f"{name}_mat") or bpy.data.materials.new(f"{name}_mat")

    for slot in CoronalPlane.material_slots:
        bpy.ops.object.material_slot_remove()
    bpy.ops.object.material_slot_add()
    CoronalPlane.active_material = mat

    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    for node in nodes:
        if node.type != "OUTPUT_MATERIAL":
            nodes.remove(node)
    ImageName = f"{name}.png"
    ImagePath = join(SlicesDir, ImageName)

    # write "2_CORONAL_SLICE.png" to here ImagePath
    BDENTAL4D_CoronalSliceUpdate(bpy.context.scene)

    BlenderImage = bpy.data.images.get(ImageName) or bpy.data.images.load(ImagePath)

    TextureCoord = AddNode(nodes, type="ShaderNodeTexCoord", name="TextureCoord")
    ImageTexture = AddNode(nodes, type="ShaderNodeTexImage", name="Image Texture")
    print(ImageTexture)
    ImageTexture.image = BlenderImage
    BlenderImage.colorspace_settings.name = "Non-Color"
    materialOutput = nodes["Material Output"]
    links.new(TextureCoord.outputs[0], ImageTexture.inputs[0])
    links.new(ImageTexture.outputs[0], materialOutput.inputs[0])
    bpy.context.scene.transform_orientation_slots[0].type = "LOCAL"
    bpy.context.scene.transform_orientation_slots[1].type = "LOCAL"
    bpy.ops.wm.tool_set_by_id(name="builtin.move")

    return CoronalPlane


def AddSagitalSlice(Preffix, DcmInfo, SlicesDir):
    
    CTVolume = bpy.context.active_object
    name = f"3_{Preffix}_SAGITAL_SLICE"
    Sp, Sz, Origin, Direction, VC = (
        DcmInfo["Spacing"],
        DcmInfo["Size"],
        DcmInfo["Origin"],
        DcmInfo["Direction"],
        DcmInfo["VolumeCenter"],
    )

    DimX, DimY, DimZ = (Sz[0] * Sp[0], Sz[1] * Sp[1], Sz[2] * Sp[2])

    # Remove old Slices and their data meshs :
    OldSlices = [obj for obj in bpy.context.view_layer.objects if name in obj.name]
    OldSliceMeshs = [mesh for mesh in bpy.data.meshes if name in mesh.name]

    for obj in OldSlices:
        bpy.data.objects.remove(obj)
    for mesh in OldSliceMeshs:
        bpy.data.meshes.remove(mesh)

    # Add SAGITAL :
    bpy.ops.mesh.primitive_plane_add()
    SagitalPlane = bpy.context.active_object
    SagitalPlane.name = name
    SagitalPlane.data.name = f"{name}_mesh"
    SagitalPlane.rotation_mode = "XYZ"
    SagitalDims = Vector((DimX, DimY, 0.0))
    SagitalPlane.dimensions = SagitalDims
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    rotation_euler = Euler((pi / 2, 0.0, -pi / 2), "XYZ")
    RotMtx = rotation_euler.to_matrix().to_4x4()
    SagitalPlane.matrix_world = CTVolume.matrix_world @ RotMtx
    # SagitalPlane.location = VC
    # Add Material :
    mat = bpy.data.materials.get(f"{name}_mat") or bpy.data.materials.new(f"{name}_mat")

    for slot in SagitalPlane.material_slots:
        bpy.ops.object.material_slot_remove()
    bpy.ops.object.material_slot_add()
    SagitalPlane.active_material = mat

    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    for node in nodes:
        if node.type != "OUTPUT_MATERIAL":
            nodes.remove(node)
    
    ImageName = f"{name}.png"
    ImagePath = join(SlicesDir, ImageName)

    # write "3_SAGITAL_SLICE.png" to here ImagePath
    BDENTAL4D_SagitalSliceUpdate(bpy.context.scene)

    BlenderImage = bpy.data.images.get(ImageName) or bpy.data.images.load(ImagePath)

    TextureCoord = AddNode(nodes, type="ShaderNodeTexCoord", name="TextureCoord")
    ImageTexture = AddNode(nodes, type="ShaderNodeTexImage", name="Image Texture")
    print(ImageTexture)
    ImageTexture.image = BlenderImage
    BlenderImage.colorspace_settings.name = "Non-Color"
    materialOutput = nodes["Material Output"]
    links.new(TextureCoord.outputs[0], ImageTexture.inputs[0])
    links.new(ImageTexture.outputs[0], materialOutput.inputs[0])
    bpy.context.scene.transform_orientation_slots[0].type = "LOCAL"
    bpy.context.scene.transform_orientation_slots[1].type = "LOCAL"
    bpy.ops.wm.tool_set_by_id(name="builtin.move")

    return SagitalPlane



@persistent
def BDENTAL4D_AxialSliceUpdate(scene):
    Planes = [
        obj
        for obj in bpy.context.scene.objects
        if ("BD4D" in obj.name and obj.name.endswith("_AXIAL_SLICE"))
    ]
    SLICES_POINTER = [
        obj
        for obj in bpy.context.scene.objects
        if ("BD4D" in obj.name and obj.name.endswith("SLICES_POINTER"))
    ]

    if Planes:
        BDENTAL_4D_Props = scene.BDENTAL_4D_Props
        ActiveObject = scene.view_layers[0].objects.active
        

        Condition1 = ActiveObject in Planes
        Condition2 = ActiveObject in SLICES_POINTER

        if Condition1:
            Preffix = ActiveObject.name[2:10]
        if Condition2:
            Preffix = ActiveObject.name[0:8]
        if Condition1 or Condition2:

            Plane = [obj for obj in Planes if Preffix in obj.name][0]
            DcmInfoDict = eval(BDENTAL_4D_Props.DcmInfo)
            DcmInfo = DcmInfoDict[Preffix]
            
            SlicesDir  = AbsPath(BDENTAL_4D_Props.SlicesDir)
            if not exists(SlicesDir) :
                SlicesDir = tempfile.mkdtemp()
                BDENTAL_4D_Props.SlicesDir = SlicesDir

            Nrrd255Path = AbsPath(DcmInfo["Nrrd255Path"])

            ImageData = Nrrd255Path

            Condition = exists(ImageData)

            if Condition:
                CTVolume = [
                    obj
                    for obj in bpy.context.scene.objects
                    if (Preffix in obj.name and "CTVolume" in obj.name)
                ][0]
                TransformMatrix = CTVolume.matrix_world

                # TransformMatrix = DcmInfo["TransformMatrix"]
                ImageName = f"{Plane.name}.png"
                ImagePath = join(SlicesDir, ImageName)

                #########################################
                #########################################
                # Get ImageData Infos :
                Image3D_255 = sitk.ReadImage(ImageData)
                Sp = Spacing = Image3D_255.GetSpacing()
                Sz = Size = Image3D_255.GetSize()
                Ortho_Origin = (
                    -0.5 * np.array(Sp) * (np.array(Sz) - np.array((1, 1, 1)))
                )
                Image3D_255.SetOrigin(Ortho_Origin)
                Image3D_255.SetDirection(np.identity(3).flatten())

                # Output Parameters :
                Out_Origin = [Ortho_Origin[0], Ortho_Origin[1], 0]
                Out_Direction = Vector(np.identity(3).flatten())
                Out_Size = (Sz[0], Sz[1], 1)
                Out_Spacing = Sp

                ######################################
                # Get Plane Orientation and location :
                PlanMatrix = TransformMatrix.inverted() @ Plane.matrix_world
                Rot = PlanMatrix.to_euler()
                Trans = PlanMatrix.translation
                Rvec = (Rot.x, Rot.y, Rot.z)
                Tvec = Trans

                ##########################################
                # Euler3DTransform :
                Euler3D = sitk.Euler3DTransform()
                Euler3D.SetCenter((0, 0, 0))
                Euler3D.SetRotation(Rvec[0], Rvec[1], Rvec[2])
                Euler3D.SetTranslation(Tvec)
                Euler3D.ComputeZYXOn()
                #########################################

                Image2D = sitk.Resample(
                    Image3D_255,
                    Out_Size,
                    Euler3D,
                    sitk.sitkLinear,
                    Out_Origin,
                    Out_Spacing,
                    Out_Direction,
                    0,
                )
                #############################################
                # Write Image :
                Array = sitk.GetArrayFromImage(Image2D)
                Flipped_Array = np.flipud(Array.reshape(Array.shape[1], Array.shape[2]))
                cv2.imwrite(ImagePath, Flipped_Array)
                #############################################
                # Update Blender Image data :
                BlenderImage = bpy.data.images.get(f"{Plane.name}.png")
                if not BlenderImage:
                    bpy.data.images.load(ImagePath)
                    BlenderImage = bpy.data.images.get(f"{Plane.name}.png")
                else:
                    BlenderImage.filepath = ImagePath
                    BlenderImage.reload()


@persistent
def BDENTAL4D_CoronalSliceUpdate(scene):

    Planes = [
        obj
        for obj in bpy.context.scene.objects
        if ("BD4D" in obj.name and obj.name.endswith("CORONAL_SLICE"))
    ]
    SLICES_POINTER = [
        obj
        for obj in bpy.context.scene.objects
        if ("BD4D" in obj.name and obj.name.endswith("SLICES_POINTER"))
    ]

    if Planes:
        BDENTAL_4D_Props = scene.BDENTAL_4D_Props
        ActiveObject = scene.view_layers[0].objects.active

        Condition1 = ActiveObject in Planes
        Condition2 = ActiveObject in SLICES_POINTER

        if Condition1:
            Preffix = ActiveObject.name[2:10]
        if Condition2:
            Preffix = ActiveObject.name[0:8]
        if Condition1 or Condition2:

            Plane = [obj for obj in Planes if Preffix in obj.name][0]
            DcmInfoDict = eval(BDENTAL_4D_Props.DcmInfo)
            DcmInfo = DcmInfoDict[Preffix]

            SlicesDir  = AbsPath(BDENTAL_4D_Props.SlicesDir)
            if not exists(SlicesDir) :
                SlicesDir = tempfile.mkdtemp()
                BDENTAL_4D_Props.SlicesDir = SlicesDir

            Nrrd255Path = AbsPath(DcmInfo["Nrrd255Path"])
            ImageData = Nrrd255Path
            Condition = exists(ImageData)

            if Condition:
                CTVolume = [
                    obj
                    for obj in bpy.context.scene.objects
                    if (Preffix in obj.name and "CTVolume" in obj.name)
                ][0]


                TransformMatrix = CTVolume.matrix_world

                # TransformMatrix = DcmInfo["TransformMatrix"]
                ImageName = f"{Plane.name}.png"
                ImagePath = join(SlicesDir, ImageName)

                #########################################
                #########################################
                # Get ImageData Infos :
                Image3D_255 = sitk.ReadImage(ImageData)
                Sp = Spacing = Image3D_255.GetSpacing()
                Sz = Size = Image3D_255.GetSize()
                Ortho_Origin = (
                    -0.5 * np.array(Sp) * (np.array(Sz) - np.array((1, 1, 1)))
                )
                Image3D_255.SetOrigin(Ortho_Origin)
                Image3D_255.SetDirection(np.identity(3).flatten())

                # Output Parameters :
                Out_Origin = [Ortho_Origin[0], Ortho_Origin[1], 0]
                Out_Direction = Vector(np.identity(3).flatten())
                Out_Size = (Sz[0], Sz[1], 1)
                Out_Spacing = Sp

                ######################################
                # Get Plane Orientation and location :
                PlanMatrix = TransformMatrix.inverted() @ Plane.matrix_world
                Rot = PlanMatrix.to_euler()
                Trans = PlanMatrix.translation
                Rvec = (Rot.x, Rot.y, Rot.z)
                Tvec = Trans

                ##########################################
                # Euler3DTransform :
                Euler3D = sitk.Euler3DTransform()
                Euler3D.SetCenter((0, 0, 0))
                Euler3D.SetRotation(Rvec[0], Rvec[1], Rvec[2])
                Euler3D.SetTranslation(Tvec)
                Euler3D.ComputeZYXOn()
                #########################################

                Image2D = sitk.Resample(
                    Image3D_255,
                    Out_Size,
                    Euler3D,
                    sitk.sitkLinear,
                    Out_Origin,
                    Out_Spacing,
                    Out_Direction,
                    0,
                )
                #############################################
                # Write Image :
                Array = sitk.GetArrayFromImage(Image2D)
                Flipped_Array = np.flipud(Array.reshape(Array.shape[1], Array.shape[2]))
                cv2.imwrite(ImagePath, Flipped_Array)
                #############################################
                # Update Blender Image data :
                BlenderImage = bpy.data.images.get(f"{Plane.name}.png")
                if not BlenderImage:
                    bpy.data.images.load(ImagePath)
                    BlenderImage = bpy.data.images.get(f"{Plane.name}.png")

                else:
                    BlenderImage.filepath = ImagePath
                    BlenderImage.reload()


@persistent
def BDENTAL4D_SagitalSliceUpdate(scene):

    Planes = [
        obj
        for obj in bpy.context.scene.objects
        if ("BD4D" in obj.name and obj.name.endswith("SAGITAL_SLICE"))
    ]
    SLICES_POINTER = [
        obj
        for obj in bpy.context.scene.objects
        if ("BD4D" in obj.name and obj.name.endswith("SLICES_POINTER"))
    ]

    if Planes:
        BDENTAL_4D_Props = scene.BDENTAL_4D_Props
        ActiveObject = scene.view_layers[0].objects.active

        Condition1 = ActiveObject in Planes
        Condition2 = ActiveObject in SLICES_POINTER

        if Condition1:
            Preffix = ActiveObject.name[2:10]
        if Condition2:
            Preffix = ActiveObject.name[0:8]
        
        if Condition1 or Condition2:

            Plane = [obj for obj in Planes if Preffix in obj.name][0]
            DcmInfoDict = eval(BDENTAL_4D_Props.DcmInfo)
            DcmInfo = DcmInfoDict[Preffix]

            SlicesDir  = AbsPath(BDENTAL_4D_Props.SlicesDir)
            if not exists(SlicesDir) :
                SlicesDir = tempfile.mkdtemp()
                BDENTAL_4D_Props.SlicesDir = SlicesDir

            Nrrd255Path = AbsPath(DcmInfo["Nrrd255Path"])
            ImageData = Nrrd255Path

            Condition = exists(ImageData)

            if Condition:
                CTVolume = [
                    obj
                    for obj in bpy.context.scene.objects
                    if (Preffix in obj.name and "CTVolume" in obj.name)
                ][0]

                TransformMatrix = CTVolume.matrix_world

                # TransformMatrix = DcmInfo["TransformMatrix"]
                ImageName = f"{Plane.name}.png"
                ImagePath = join(SlicesDir, ImageName)

                #########################################
                #########################################
                # Get ImageData Infos :
                Image3D_255 = sitk.ReadImage(ImageData)
                Sp = Spacing = Image3D_255.GetSpacing()
                Sz = Size = Image3D_255.GetSize()
                Ortho_Origin = (
                    -0.5 * np.array(Sp) * (np.array(Sz) - np.array((1, 1, 1)))
                )
                Image3D_255.SetOrigin(Ortho_Origin)
                Image3D_255.SetDirection(np.identity(3).flatten())

                # Output Parameters :
                Out_Origin = [Ortho_Origin[0], Ortho_Origin[1], 0]
                Out_Direction = Vector(np.identity(3).flatten())
                Out_Size = (Sz[0], Sz[1], 1)
                Out_Spacing = Sp

                ######################################
                # Get Plane Orientation and location :
                PlanMatrix = TransformMatrix.inverted() @ Plane.matrix_world
                Rot = PlanMatrix.to_euler()
                Trans = PlanMatrix.translation
                Rvec = (Rot.x, Rot.y, Rot.z)
                Tvec = Trans

                ##########################################
                # Euler3DTransform :
                Euler3D = sitk.Euler3DTransform()
                Euler3D.SetCenter((0, 0, 0))
                Euler3D.SetRotation(Rvec[0], Rvec[1], Rvec[2])
                Euler3D.SetTranslation(Tvec)
                Euler3D.ComputeZYXOn()
                #########################################

                Image2D = sitk.Resample(
                    Image3D_255,
                    Out_Size,
                    Euler3D,
                    sitk.sitkLinear,
                    Out_Origin,
                    Out_Spacing,
                    Out_Direction,
                    0,
                )
                #############################################
                # Write Image :
                Array = sitk.GetArrayFromImage(Image2D)
                Flipped_Array = np.flipud(Array.reshape(Array.shape[1], Array.shape[2]))
                cv2.imwrite(ImagePath, Flipped_Array)
                #############################################
                # Update Blender Image data :
                BlenderImage = bpy.data.images.get(f"{Plane.name}.png")
                if not BlenderImage:
                    bpy.data.images.load(ImagePath)
                    BlenderImage = bpy.data.images.get(f"{Plane.name}.png")
                else:
                    BlenderImage.filepath = ImagePath
                    BlenderImage.reload()


####################################################################
def Add_Cam_To_Plane(Plane, CamDistance, ClipOffset):
    Override, _, _ = CtxOverride(bpy.context)
    bpy.ops.object.camera_add(Override)
    Cam = bpy.context.object
    Cam.name = f"{Plane.name}_CAM"
    Cam.data.name = f"{Plane.name}_CAM_data"
    Cam.data.type = "ORTHO"
    Cam.data.ortho_scale = max(Plane.dimensions) * 1.1
    Cam.data.display_size = 10

    Cam.matrix_world = Plane.matrix_world
    bpy.ops.transform.translate(
        value=(0, 0, CamDistance),
        orient_type="LOCAL",
        orient_matrix=Plane.matrix_world.to_3x3(),
        orient_matrix_type="LOCAL",
        constraint_axis=(False, False, True),
    )
    Cam.data.clip_start = CamDistance - ClipOffset
    Cam.data.clip_end = CamDistance + ClipOffset

    Plane.select_set(True)
    bpy.context.view_layer.objects.active = Plane
    bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)
    Cam.hide_set(True)
    Cam.select_set(False)
    return Cam


# def Add_Cam_To_Plane(Plane, CamView, Override, ActiveSpace):

#     bpy.ops.object.camera_add(Override)
#     Cam = bpy.context.object
#     Cam.name = f'BDENTAL_4D_CAM_{CamView}'
#     Cam.data.name = f'BDENTAL_4D_CAM_{CamView}_data'
#     Cam.data.type = 'ORTHO'
#     Cam.data.ortho_scale = max(Plane.dimensions)*1.1
#     Cam.data.display_size = 10

#     Cam.matrix_world = Plane.matrix_world

#     bpy.ops.transform.translate(Override, value=(0, 0, 100), orient_type='LOCAL', orient_matrix=Plane.matrix_world.to_3x3(), orient_matrix_type='LOCAL', constraint_axis=(False, False, True))
#     Active_Space.camera = Cam
#     bpy.ops.view3d.view_camera(Override)
#     Plane.select_set(True)
#     bpy.context.view_layer.objects.active = Plane
#     bpy.ops.object.parent_set(Override, type='OBJECT', keep_transform=True)
#     Cam.hide_set(True)
#     Cam.select_set(False)

#############################################################################
# SimpleITK vtk Image to Mesh Functions :
#############################################################################
def HuTo255(Hu, Wmin, Wmax):
    V255 = int(((Hu - Wmin) / (Wmax - Wmin)) * 255)
    return V255

def vtkWindowedSincPolyDataFilter(q, mesh, Iterations, step, start, finish) :
    def VTK_Terminal_progress(caller, event):
        ProgRatio = round(float(caller.GetProgress()), 2)
        q.put(
            [
                "loop",
                f"PROGRESS : {step}...",
                "",
                start,
                finish,
                ProgRatio,
            ]
        )
    SmoothFilter = vtk.vtkWindowedSincPolyDataFilter()
    SmoothFilter.SetInputData(mesh)
    SmoothFilter.SetNumberOfIterations(Iterations)
    SmoothFilter.BoundarySmoothingOff()
    SmoothFilter.FeatureEdgeSmoothingOn()
    SmoothFilter.SetFeatureAngle(60)
    SmoothFilter.SetPassBand(0.01)
    SmoothFilter.NonManifoldSmoothingOn()
    SmoothFilter.NormalizeCoordinatesOn()
    SmoothFilter.AddObserver(ProgEvent, VTK_Terminal_progress)
    SmoothFilter.Update()
    mesh.DeepCopy(SmoothFilter.GetOutput())
    return mesh

def ResizeImage(sitkImage, Ratio):
    image = sitkImage
    Sz = image.GetSize()
    Sp = image.GetSpacing()
    new_size = [int(Sz[0] * Ratio), int(Sz[1] * Ratio), int(Sz[2] * Ratio)]
    new_spacing = [Sp[0] / Ratio, Sp[1] / Ratio, Sp[2] / Ratio]

    ResizedImage = sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
        0,
    )
    return ResizedImage , new_size, new_spacing


# def VTK_Terminal_progress(caller, event, q):
#     ProgRatio = round(float(caller.GetProgress()), 2)
#     q.put(
#         ["loop", f"PROGRESS : {step} processing...", "", {start}, {finish}, ProgRatio]
#     )


def VTKprogress(caller, event):
    pourcentage = int(caller.GetProgress() * 100)
    calldata = str(int(caller.GetProgress() * 100)) + " %"
    # print(calldata)
    sys.stdout.write(f"\r {calldata}")
    sys.stdout.flush()
    progress_bar(pourcentage, Delay=1)


def TerminalProgressBar(
    q,
    counter_start,
    iter=100,
    maxfill=20,
    symb1="\u2588",
    symb2="\u2502",
    periode=10,
):

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="cp65001")
        # cmd = "chcp 65001 & set PYTHONIOENCODING=utf-8"
        # subprocess.call(cmd, shell=True)

    print("\n")

    while True:
        if not q.empty():
            signal = q.get()

            if "End" in signal[0]:
                finish = Tcounter()
                line = f"{symb1*maxfill}  100% Finished.------Total Time : {round(finish-counter_start,2)}"
                # clear sys.stdout line and return to line start:
                # sys.stdout.write("\r")
                # sys.stdout.write(" " * 100)
                # sys.stdout.flush()
                # sys.stdout.write("\r")
                # write line :
                sys.stdout.write("\r" + " " * 80 + "\r" + line)  # f"{Char}"*i*2
                sys.stdout.flush()
                break

            if "GuessTime" in signal[0]:
                _, Uptxt, Lowtxt, start, finish, periode = signal
                for i in range(iter):

                    if q.empty():

                        ratio = start + (((i + 1) / iter) * (finish - start))
                        pourcentage = int(ratio * 100)
                        symb1_fill = int(ratio * maxfill)
                        symb2_fill = int(maxfill - symb1_fill)
                        line = f"{symb1*symb1_fill}{symb2*symb2_fill}  {pourcentage}% {Uptxt}"
                        # clear sys.stdout line and return to line start:
                        # sys.stdout.write("\r"+" " * 80)
                        # sys.stdout.flush()
                        # write line :
                        sys.stdout.write("\r" + " " * 80 + "\r" + line)  # f"{Char}"*i*2
                        sys.stdout.flush()
                        sleep(periode / iter)
                    else:
                        break

            if "loop" in signal[0]:
                _, Uptxt, Lowtxt, start, finish, progFloat = signal
                ratio = start + (progFloat * (finish - start))
                pourcentage = int(ratio * 100)
                symb1_fill = int(ratio * maxfill)
                symb2_fill = int(maxfill - symb1_fill)
                line = f"{symb1*symb1_fill}{symb2*symb2_fill}  {pourcentage}% {Uptxt}"
                # clear sys.stdout line and return to line start:
                # sys.stdout.write("\r")
                # sys.stdout.write(" " * 100)
                # sys.stdout.flush()
                # sys.stdout.write("\r")
                # write line :
                sys.stdout.write("\r" + " " * 80 + "\r" + line)  # f"{Char}"*i*2
                sys.stdout.flush()

        else:
            sleep(0.1)


def sitkTovtk(sitkImage):
    """Convert sitk image to a VTK image"""
    sitkArray = sitk.GetArrayFromImage(sitkImage)  # .astype(np.uint8)
    vtkImage = vtk.vtkImageData()

    Sp = Spacing = sitkImage.GetSpacing()
    Sz = Size = sitkImage.GetSize()

    vtkImage.SetDimensions(Sz)
    vtkImage.SetSpacing(Sp)
    vtkImage.SetOrigin(0, 0, 0)
    vtkImage.SetDirectionMatrix(1, 0, 0, 0, 1, 0, 0, 0, 1)
    vtkImage.SetExtent(0, Sz[0] - 1, 0, Sz[1] - 1, 0, Sz[2] - 1)

    VtkArray = numpy_support.numpy_to_vtk(
        sitkArray.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_INT
    )
    VtkArray.SetNumberOfComponents(1)
    vtkImage.GetPointData().SetScalars(VtkArray)

    vtkImage.Modified()
    return vtkImage


def vtk_MC_Func(vtkImage, Treshold):
    MCFilter = vtk.vtkMarchingCubes()
    MCFilter.ComputeNormalsOn()
    MCFilter.SetValue(0, Treshold)
    MCFilter.SetInputData(vtkImage)
    MCFilter.Update()
    mesh = vtk.vtkPolyData()
    mesh.DeepCopy(MCFilter.GetOutput())
    return mesh


def vtkMeshReduction(q, mesh, reduction, step, start, finish):
    """Reduce a mesh using VTK's vtkQuadricDecimation filter."""

    def VTK_Terminal_progress(caller, event):
        ProgRatio = round(float(caller.GetProgress()), 2)
        q.put(
            [
                "loop",
                f"PROGRESS : {step}...",
                "",
                start,
                finish,
                ProgRatio,
            ]
        )

    decimatFilter = vtk.vtkQuadricDecimation()
    decimatFilter.SetInputData(mesh)
    decimatFilter.SetTargetReduction(reduction)

    decimatFilter.AddObserver(ProgEvent, VTK_Terminal_progress)
    decimatFilter.Update()

    mesh.DeepCopy(decimatFilter.GetOutput())
    return mesh


def vtkSmoothMesh(q, mesh, Iterations, step, start, finish):
    """Smooth a mesh using VTK's vtkSmoothPolyData filter."""

    def VTK_Terminal_progress(caller, event):
        ProgRatio = round(float(caller.GetProgress()), 2)
        q.put(
            [
                "loop",
                f"PROGRESS : {step}...",
                "",
                start,
                finish,
                ProgRatio,
            ]
        )

    SmoothFilter = vtk.vtkSmoothPolyDataFilter()
    SmoothFilter.SetInputData(mesh)
    SmoothFilter.SetNumberOfIterations(int(Iterations))
    SmoothFilter.SetFeatureAngle(45)
    SmoothFilter.SetRelaxationFactor(0.05)
    SmoothFilter.AddObserver(ProgEvent, VTK_Terminal_progress)
    SmoothFilter.Update()
    mesh.DeepCopy(SmoothFilter.GetOutput())
    return mesh


def vtkTransformMesh(mesh, Matrix):
    """Transform a mesh using VTK's vtkTransformPolyData filter."""

    Transform = vtk.vtkTransform()
    Transform.SetMatrix(Matrix)

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(mesh)
    transformFilter.SetTransform(Transform)
    transformFilter.Update()
    mesh.DeepCopy(transformFilter.GetOutput())
    return mesh


def vtkfillholes(mesh, size):
    FillHolesFilter = vtk.vtkFillHolesFilter()
    FillHolesFilter.SetInputData(mesh)
    FillHolesFilter.SetHoleSize(size)
    FillHolesFilter.Update()
    mesh.DeepCopy(FillHolesFilter.GetOutput())
    return mesh


def vtkCleanMesh(mesh, connectivityFilter=False):
    """Clean a mesh using VTK's CleanPolyData filter."""

    ConnectFilter = vtk.vtkPolyDataConnectivityFilter()
    CleanFilter = vtk.vtkCleanPolyData()

    if connectivityFilter:

        ConnectFilter.SetInputData(mesh)
        ConnectFilter.SetExtractionModeToLargestRegion()
        CleanFilter.SetInputConnection(ConnectFilter.GetOutputPort())

    else:
        CleanFilter.SetInputData(mesh)

    CleanFilter.Update()
    mesh.DeepCopy(CleanFilter.GetOutput())
    return mesh


def sitkToContourArray(sitkImage, HuMin, HuMax, Wmin, Wmax, Thikness):
    """Convert sitk image to a VTK image"""

    def HuTo255(Hu, Wmin, Wmax):
        V255 = ((Hu - Wmin) / (Wmax - Wmin)) * 255
        return V255

    Image3D_255 = sitk.Cast(
        sitk.IntensityWindowing(
            sitkImage,
            windowMinimum=Wmin,
            windowMaximum=Wmax,
            outputMinimum=0.0,
            outputMaximum=255.0,
        ),
        sitk.sitkUInt8,
    )
    Array = sitk.GetArrayFromImage(Image3D_255)
    ContourArray255 = Array.copy()
    for i in range(ContourArray255.shape[0]):
        Slice = ContourArray255[i, :, :]
        ret, binary = cv2.threshold(
            Slice,
            HuTo255(HuMin, Wmin, Wmax),
            HuTo255(HuMax, Wmin, Wmax),
            cv2.THRESH_BINARY,
        )
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        SliceContour = np.ones(binary.shape, dtype="uint8")
        cv2.drawContours(SliceContour, contours, -1, 255, Thikness)

        ContourArray255[i, :, :] = SliceContour

    return ContourArray255


def vtkContourFilter(vtkImage, isovalue=0.0):
    """Extract an isosurface from a volume."""

    ContourFilter = vtk.vtkContourFilter()
    ContourFilter.SetInputData(vtkImage)
    ContourFilter.SetValue(0, isovalue)
    ContourFilter.Update()
    mesh = vtk.vtkPolyData()
    mesh.DeepCopy(ContourFilter.GetOutput())
    return mesh


def GuessTimeLoopFunc(signal, q):
    _, Uptxt, Lowtxt, start, finish, periode = signal
    i = 0
    iterations = 10
    while i < iterations and q.empty():
        ProgRatio = start + (((i + 1) / iterations) * (finish - start))
        q.put(
            [
                "loop",
                Uptxt,
                "",
                start,
                finish,
                ProgRatio,
            ]
        )
        sleep(periode / iterations)
        i += 1


def CV2_progress_bar(q, iter=100):
    while True:
        if not q.empty():
            signal = q.get()

            if "End" in signal[0]:
                pourcentage = 100
                Uptxt = "Finished."
                progress_bar(pourcentage, Uptxt)
                break
            if "GuessTime" in signal[0]:
                _, Uptxt, Lowtxt, start, finish, periode = signal
                i = 0
                iterations = 10
                Delay = periode / iterations
                while i < iterations:
                    if q.empty():
                        ProgRatio = start + (
                            round(((i + 1) / iterations), 2) * (finish - start)
                        )
                        pourcentage = int(ProgRatio * 100)
                        progress_bar(
                            pourcentage, Uptxt, Delay=int(Delay * 1000)
                        )  # , Delay = int(Delay*1000)
                        sleep(Delay)
                        i += 1
                    else:
                        break
                # t = threading.Thread(target=GuessTimeLoopFunc, args=[signal, q], daemon=True)
                # t.start()
                # t.join()
                # while i < iterations and q.empty() :
                #     ratio = start + (((i + 1) / iter) * (finish - start))
                #     pourcentage = int(ratio * 100)
                #     progress_bar(pourcentage, Uptxt)
                #     sleep(periode / iter)

                # iter = 5
                # _, Uptxt, Lowtxt, start, finish, periode = signal
                # for i in range(iter):

                #     if q.empty():

                #         ratio = start + (((i + 1) / iter) * (finish - start))
                #         pourcentage = int(ratio * 100)
                #         progress_bar(pourcentage, Uptxt)
                #         sleep(periode / iter)
                #     else:
                #         break

            if "loop" in signal[0]:
                _, Uptxt, Lowtxt, start, finish, progFloat = signal
                ratio = start + (progFloat * (finish - start))
                pourcentage = int(ratio * 100)
                progress_bar(pourcentage, Uptxt)

        else:
            sleep(0.01)


def progress_bar(pourcentage, Uptxt, Lowtxt="", Title="BDENTAL_4D", Delay=1):

    X, Y = WindowWidth, WindowHeight = (500, 100)
    BackGround = np.ones((Y, X, 3), dtype=np.uint8) * 255
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

    chunk = (maxFill - 40) / 100

    img = BackGround.copy()
    fill = minFill + int(pourcentage * chunk)
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
        Uptxt,
        (minFill, barUp - 10),
        font,
        fontScale,
        fontColor,
        fontThikness,
        lineStyle,
    )
    cv2.imshow(Title, img)

    cv2.waitKey(Delay)

    if pourcentage == 100:
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
            Uptxt,
            (minFill, barUp - 10),
            font,
            fontScale,
            fontColor,
            fontThikness,
            lineStyle,
        )
        cv2.imshow(Title, img)
        cv2.waitKey(Delay)
        sleep(4)
        cv2.destroyAllWindows()


######################################################
# BDENTAL_4D Meshes Tools Operators...........
######################################################
def AddCurveSphere(Name, Curve, i, CollName):
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    bezier_points = Curve.data.splines[0].bezier_points[:]
    Bpt = bezier_points[i]
    loc = Curve.matrix_world @ Bpt.co
    AddMarkupPoint(
        name=Name, color=(0, 1, 0, 1), loc=loc, Diameter=0.5, CollName=CollName
    )
    Hook = bpy.context.object
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    Hook.select_set(True)
    Curve.select_set(True)
    bpy.context.view_layer.objects.active = Curve
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.curve.select_all(action="DESELECT")
    bezier_points = Curve.data.splines[0].bezier_points[:]
    Bpt = bezier_points[i]
    Bpt.select_control_point = True
    bpy.ops.object.hook_add_selob(use_bone=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    Curve.select_set(True)
    bpy.context.view_layer.objects.active = Curve

    return Hook


def CuttingCurveAdd():

    # Prepare scene settings :
    bpy.ops.transform.select_orientation(orientation="GLOBAL")
    bpy.context.scene.tool_settings.use_snap = True
    bpy.context.scene.tool_settings.snap_elements = {"FACE"}
    bpy.context.scene.tool_settings.transform_pivot_point = "INDIVIDUAL_ORIGINS"

    # Get CuttingTarget :
    CuttingTargetName = bpy.context.scene.BDENTAL_4D_Props.CuttingTargetNameProp
    CuttingTarget = bpy.data.objects[CuttingTargetName]
    # ....Add Curve ....... :
    bpy.ops.curve.primitive_bezier_curve_add(
        radius=1, enter_editmode=False, align="CURSOR"
    )
    # Set cutting_tool name :
    CurveCutter = bpy.context.view_layer.objects.active
    CurveCutter.name = "BDENTAL4D_Curve_Cut1"
    curve = CurveCutter.data
    curve.name = "BDENTAL4D_Curve_Cut1"
    bpy.context.scene.BDENTAL_4D_Props.CurveCutterNameProp = CurveCutter.name
    MoveToCollection(CurveCutter, "BDENTAL-4D Cutters")
    # CurveCutter settings :
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.curve.select_all(action="DESELECT")
    curve.splines[0].bezier_points[-1].select_control_point = True
    bpy.ops.curve.dissolve_verts()
    B0_Point = curve.splines[0].bezier_points[0]
    B0_Point.select_control_point = True
    # bpy.ops.curve.select_all(action="SELECT")
    bpy.ops.view3d.snap_selected_to_cursor(use_offset=False)

    bpy.context.object.data.dimensions = "3D"
    bpy.context.object.data.twist_smooth = 4
    bpy.ops.curve.handle_type_set(type="AUTOMATIC")
    bpy.context.object.data.bevel_depth = 0.2

    bpy.context.object.data.bevel_resolution = 10
    bpy.context.scene.tool_settings.curve_paint_settings.error_threshold = 1
    bpy.context.scene.tool_settings.curve_paint_settings.corner_angle = 0.785398
    # bpy.context.scene.tool_settings.curve_paint_settings.corner_angle = 1.5708
    bpy.context.scene.tool_settings.curve_paint_settings.depth_mode = "SURFACE"
    bpy.context.scene.tool_settings.curve_paint_settings.surface_offset = 0
    bpy.context.scene.tool_settings.curve_paint_settings.use_offset_absolute = True

    # Add color material :
    CurveCutterMat = bpy.data.materials.get(
        "BDENTAL4D_Curve_Cut1_Mat"
    ) or bpy.data.materials.new("BDENTAL4D_Curve_Cut1_Mat")
    CurveCutterMat.diffuse_color = [0.1, 0.4, 1.0, 1.0]
    CurveCutterMat.roughness = 0.3

    curve.materials.append(CurveCutterMat)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.wm.tool_set_by_id(name="builtin.cursor")
    bpy.context.space_data.overlay.show_outline_selected = False

    # bpy.ops.object.modifier_add(type="SHRINKWRAP")
    # bpy.context.object.modifiers["Shrinkwrap"].target = CuttingTarget
    # bpy.context.object.modifiers["Shrinkwrap"].wrap_mode = "ABOVE_SURFACE"
    # bpy.context.object.modifiers["Shrinkwrap"].use_apply_on_spline = True


#######################################################################################
def AddTube(context, CuttingTarget):

    BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props
    # Prepare scene settings :
    bpy.ops.transform.select_orientation(orientation="GLOBAL")
    bpy.context.scene.tool_settings.use_snap = True
    bpy.context.scene.tool_settings.snap_elements = {"FACE"}
    bpy.context.scene.tool_settings.transform_pivot_point = "INDIVIDUAL_ORIGINS"

    # ....Add Curve ....... :
    bpy.ops.curve.primitive_bezier_curve_add(
        radius=1, enter_editmode=False, align="CURSOR"
    )
    # Set cutting_tool name :
    TubeObject = context.view_layer.objects.active
    TubeObject.name = "BDENTAL4D_Tube"
    TubeData = TubeObject.data
    TubeData.name = "BDENTAL4D_Tube"

    # Tube settings :
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.curve.select_all(action="DESELECT")
    TubeData.splines[0].bezier_points[-1].select_control_point = True
    bpy.ops.curve.dissolve_verts()
    bpy.ops.curve.select_all(action="SELECT")
    bpy.ops.view3d.snap_selected_to_cursor()

    TubeData.dimensions = "3D"
    TubeData.twist_smooth = 3
    bpy.ops.curve.handle_type_set(type="AUTOMATIC")
    TubeData.bevel_depth = BDENTAL_4D_Props.TubeWidth
    TubeData.bevel_resolution = 10
    bpy.context.scene.tool_settings.curve_paint_settings.error_threshold = 1
    bpy.context.scene.tool_settings.curve_paint_settings.corner_angle = 0.785398
    bpy.context.scene.tool_settings.curve_paint_settings.depth_mode = "SURFACE"
    bpy.context.scene.tool_settings.curve_paint_settings.surface_offset = 0
    bpy.context.scene.tool_settings.curve_paint_settings.use_offset_absolute = True

    # Add color material :
    TubeMat = bpy.data.materials.get("BDENTAL4D_Tube_Mat") or bpy.data.materials.new(
        "BDENTAL4D_Tube_Mat"
    )
    TubeMat.diffuse_color = [0.03, 0.20, 0.14, 1.0]  # [0.1, 0.4, 1.0, 1.0]
    TubeMat.roughness = 0.3

    TubeObject.active_material = TubeMat
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.wm.tool_set_by_id(name="builtin.cursor")
    bpy.context.space_data.overlay.show_outline_selected = False

    bpy.ops.object.modifier_add(type="SHRINKWRAP")
    bpy.context.object.modifiers["Shrinkwrap"].target = CuttingTarget
    bpy.context.object.modifiers["Shrinkwrap"].wrap_mode = "ABOVE_SURFACE"
    bpy.context.object.modifiers["Shrinkwrap"].use_apply_on_spline = True

    return TubeObject


def DeleteTubePoint(TubeObject):
    bpy.ops.object.mode_set(mode="OBJECT")

    TubeData = TubeObject.data

    try:
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.curve.select_all(action="DESELECT")
        points = TubeData.splines[0].bezier_points[:]
        points[-1].select_control_point = True
        points = TubeData.splines[0].bezier_points[:]
        if len(points) > 1:
            bpy.ops.curve.delete(type="VERT")
            points = TubeData.splines[0].bezier_points[:]
            bpy.ops.curve.select_all(action="SELECT")
            bpy.ops.curve.handle_type_set(type="AUTOMATIC")
            bpy.ops.curve.select_all(action="DESELECT")
            points = TubeData.splines[0].bezier_points[:]
            points[-1].select_control_point = True

        bpy.ops.object.mode_set(mode="OBJECT")

    except Exception:
        pass


def ExtrudeTube(TubeObject):

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.curve.extrude(mode="INIT")
    bpy.ops.view3d.snap_selected_to_cursor()
    bpy.ops.curve.select_all(action="SELECT")
    bpy.ops.curve.handle_type_set(type="AUTOMATIC")
    bpy.ops.curve.select_all(action="DESELECT")
    points = TubeObject.data.splines[0].bezier_points[:]
    points[-1].select_control_point = True
    bpy.ops.object.mode_set(mode="OBJECT")


#######################################################################################


def CuttingCurveAdd2():
    # Prepare scene settings :
    bpy.ops.transform.select_orientation(orientation="GLOBAL")
    bpy.context.scene.tool_settings.use_snap = True
    bpy.context.scene.tool_settings.snap_elements = {"FACE"}
    bpy.context.scene.tool_settings.transform_pivot_point = "INDIVIDUAL_ORIGINS"

    # Get CuttingTarget :
    CuttingTargetName = bpy.context.scene.BDENTAL_4D_Props.CuttingTargetNameProp
    CuttingTarget = bpy.data.objects[CuttingTargetName]
    # ....Add Curve ....... :
    bpy.ops.curve.primitive_bezier_curve_add(
        radius=1, enter_editmode=False, align="CURSOR"
    )
    # Set cutting_tool name :
    CurveCutter = bpy.context.view_layer.objects.active
    CurveCutter.name = "BDENTAL4D_Curve_Cut2"
    curve = CurveCutter.data
    curve.name = "BDENTAL4D_Curve_Cut2"
    bpy.context.scene.BDENTAL_4D_Props.CurveCutterNameProp = CurveCutter.name

    # CurveCutter settings :
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.curve.select_all(action="DESELECT")
    curve.splines[0].bezier_points[-1].select_control_point = True
    bpy.ops.curve.dissolve_verts()
    bpy.ops.curve.select_all(action="SELECT")
    bpy.ops.view3d.snap_selected_to_cursor(use_offset=False)

    bpy.context.object.data.dimensions = "3D"
    bpy.context.object.data.twist_smooth = 3
    bpy.ops.curve.handle_type_set(type="AUTOMATIC")
    bpy.context.object.data.bevel_depth = 0.1
    bpy.context.object.data.bevel_resolution = 6
    bpy.context.scene.tool_settings.curve_paint_settings.error_threshold = 1
    bpy.context.scene.tool_settings.curve_paint_settings.corner_angle = 0.785398
    # bpy.context.scene.tool_settings.curve_paint_settings.corner_angle = 1.5708
    bpy.context.scene.tool_settings.curve_paint_settings.depth_mode = "SURFACE"
    bpy.context.scene.tool_settings.curve_paint_settings.surface_offset = 0
    bpy.context.scene.tool_settings.curve_paint_settings.use_offset_absolute = True

    # Add color material :
    CurveCutterMat = bpy.data.materials.get(
        "BDENTAL4D_Curve_Cut2_Mat"
    ) or bpy.data.materials.new("BDENTAL4D_Curve_Cut2_Mat")
    CurveCutterMat.diffuse_color = [0.1, 0.4, 1.0, 1.0]
    CurveCutterMat.roughness = 0.3

    curve.materials.append(CurveCutterMat)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.wm.tool_set_by_id(name="builtin.cursor")
    bpy.context.space_data.overlay.show_outline_selected = False

    bpy.ops.object.modifier_add(type="SHRINKWRAP")
    bpy.context.object.modifiers["Shrinkwrap"].target = CuttingTarget
    bpy.context.object.modifiers["Shrinkwrap"].wrap_mode = "ABOVE_SURFACE"
    bpy.context.object.modifiers["Shrinkwrap"].use_apply_on_spline = True

    MoveToCollection(CurveCutter, "BDENTAL-4D Cutters")

    return CurveCutter


#######################################################################################
def DeleteLastCurvePoint():
    bpy.ops.object.mode_set(mode="OBJECT")
    # Get CuttingTarget :
    CuttingTargetName = bpy.context.scene.BDENTAL_4D_Props.CuttingTargetNameProp
    CuttingTarget = bpy.data.objects[CuttingTargetName]

    # Get CurveCutter :
    CurveCutterName = bpy.context.scene.BDENTAL_4D_Props.CurveCutterNameProp
    CurveCutter = bpy.data.objects[CurveCutterName]
    curve = CurveCutter.data
    points = curve.splines[0].bezier_points[:]
    try:
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.curve.select_all(action="DESELECT")
        points = curve.splines[0].bezier_points[:]
        points[-1].select_control_point = True
        points = curve.splines[0].bezier_points[:]
        if len(points) > 1:

            bpy.ops.curve.delete(type="VERT")
            points = curve.splines[0].bezier_points[:]
            bpy.ops.curve.select_all(action="SELECT")
            bpy.ops.curve.handle_type_set(type="AUTOMATIC")
            bpy.ops.curve.select_all(action="DESELECT")
            points = curve.splines[0].bezier_points[:]
            points[-1].select_control_point = True

        bpy.ops.object.mode_set(mode="OBJECT")

    except Exception:
        pass


#######################################################################################
def ExtrudeCurvePointToCursor(context, event):

    # Get CurveCutter :
    CurveCutterName = bpy.context.scene.BDENTAL_4D_Props.CurveCutterNameProp
    CurveCutter = bpy.data.objects[CurveCutterName]
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.curve.extrude(mode="INIT")
    bpy.ops.view3d.snap_selected_to_cursor(use_offset=False)
    bpy.ops.curve.select_all(action="SELECT")
    bpy.ops.curve.handle_type_set(type="AUTOMATIC")
    bpy.ops.curve.select_all(action="DESELECT")
    points = CurveCutter.data.splines[0].bezier_points[:]
    points[-1].select_control_point = True
    bpy.ops.object.mode_set(mode="OBJECT")


#######################################################################################
# 1st separate method function :
def SplitSeparator(CuttingTarget):
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    intersect_vgroup = CuttingTarget.vertex_groups["intersect_vgroup"]
    CuttingTarget.vertex_groups.active_index = intersect_vgroup.index
    bpy.ops.object.vertex_group_select()

    bpy.ops.mesh.edge_split()

    # Separate by loose parts :
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.mesh.separate(type="LOOSE")


#######################################################################################
# 2nd separate method function :
def IterateSeparator():

    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.select_all(action="SELECT")
    selected_initial = bpy.context.selected_objects[:]
    bpy.ops.object.select_all(action="DESELECT")
    # VisObj = bpy.context.visible_objects[:].copy()

    for obj in selected_initial:

        try:
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj

            # Select intesecting vgroup + more :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.select_all(action="DESELECT")
            intersect_vgroup = obj.vertex_groups["intersect_vgroup"]
            obj.vertex_groups.active_index = intersect_vgroup.index
            bpy.ops.object.vertex_group_select()
            # bpy.ops.mesh.select_more()

            # Get selected unselected verts :

            mesh = obj.data
            # polys = mesh.polygons
            verts = mesh.vertices
            # Polys = mesh.polygons
            # bpy.context.tool_settings.mesh_select_mode = (True, False, False)
            bpy.ops.object.mode_set(mode="OBJECT")
            # unselected_polys = [p.index for p in Polys if p.select == False]
            unselected_verts = [v.index for v in verts if v.select == False]

            # Hide intesecting vgroup :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.hide(unselected=False)

            # select a part :
            bpy.ops.object.mode_set(mode="OBJECT")
            verts[unselected_verts[0]].select = True
            bpy.ops.object.mode_set(mode="EDIT")

            bpy.ops.mesh.select_linked(delimit=set())
            bpy.ops.mesh.reveal()

            # ....Separate by selection.... :
            bpy.ops.mesh.separate(type="SELECTED")
            bpy.ops.object.mode_set(mode="OBJECT")

        except Exception:
            pass
    resulting_parts = PartsFilter()  # all visible objects are selected after func

    if resulting_parts == len(selected_initial):
        return False
    else:
        return True


#######################################################################################
# Filter loose parts function :
def PartsFilter():

    # Filter small parts :
    VisObj = bpy.context.visible_objects[:].copy()
    ObjToRemove = []
    for obj in VisObj:
        if not obj.data.polygons:
            ObjToRemove.append(obj)
        else:
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            verts = obj.data.vertices
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="DESELECT")
            bpy.ops.mesh.select_non_manifold()
            bpy.ops.mesh.remove_doubles()
            bpy.ops.object.mode_set(mode="OBJECT")
            non_manifold_verts = [v for v in verts if v.select == True]

            if len(verts) < len(non_manifold_verts) * 2:
                ObjToRemove.append(obj)

    # Remove small parts :
    for obj in ObjToRemove:
        bpy.data.objects.remove(obj)

    bpy.ops.object.select_all(action="SELECT")
    # resulting_parts = len(bpy.context.selected_objects)

    # return resulting_parts


#######################################################################################
# CurveCutter 2 functions :
#######################################################################################
def CutterPointsList(cutter, obj):

    curve = cutter.data
    CurveCoList = []
    for point in curve.splines[0].bezier_points:
        p_co_global = cutter.matrix_world @ point.co
        p_co_obj_relative = obj.matrix_world.inverted() @ p_co_global
        CurveCoList.append(p_co_obj_relative)

    return CurveCoList


def ClosestVerts(i, CurveCoList, obj):

    # initiate a KDTree :
    size = len(obj.data.vertices)
    kd = kdtree.KDTree(size)

    for v_id, v in enumerate(obj.data.vertices):
        kd.insert(v.co, v_id)

    kd.balance()
    v_co, v_id, dist = kd.find(CurveCoList[i])

    return v_id


def ClosestVertToPoint(Point, obj):

    # initiate a KDTree :
    size = len(obj.data.vertices)
    kd = kdtree.KDTree(size)

    for v_id, v in enumerate(obj.data.vertices):
        kd.insert(v.co, v_id)

    kd.balance()
    v_co, v_id, dist = kd.find(Point)

    return v_id, v_co, dist


# Add square cutter function :
def add_square_cutter(context):

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="MEDIAN")

    Model = bpy.context.view_layer.objects.active
    loc = Model.location.copy()  # get model location
    view_rotation = context.space_data.region_3d.view_rotation

    view3d_rot_matrix = (
        view_rotation.to_matrix().to_4x4()
    )  # get v3d rotation matrix 4x4

    # Add cube :
    bpy.ops.mesh.primitive_cube_add(size=120, enter_editmode=False)

    frame = bpy.context.view_layer.objects.active
    for obj in bpy.data.objects:
        if obj.name == "my_frame_cutter":
            obj.name = "my_frame_cutter_old"
    frame.name = "my_frame_cutter"

    # Reshape and align cube :

    frame.matrix_world = view3d_rot_matrix

    frame.location = loc

    bpy.context.object.display_type = "WIRE"
    bpy.context.object.scale[1] = 0.5
    bpy.context.object.scale[2] = 2

    # Subdivide cube 10 iterations 3 times :

    bpy.ops.object.select_all(action="DESELECT")
    frame.select_set(True)
    bpy.context.view_layer.objects.active = frame

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.subdivide(number_cuts=10)
    bpy.ops.mesh.subdivide(number_cuts=6)

    # Make cube normals consistent :

    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.mesh.select_all(action="DESELECT")

    bpy.ops.object.mode_set(mode="OBJECT")

    # Select frame :

    bpy.ops.object.select_all(action="DESELECT")
    frame.select_set(True)
    bpy.context.view_layer.objects.active = frame


###########################################################################
# Add BDENTAL_4D MultiView :
def getLocalCollIndex(collName) :
    assert bpy.data.collections
    collNames = [col.name for col in bpy.context.scene.collection.children]
    
    if collName in collNames :
        index = collNames.index(collName)
        return index+1
    else : return None
def BDENTAL_4D_MultiView_Toggle(Preffix):
    COLLS = bpy.context.view_layer.layer_collection.children
    collectionState = {col:col.hide_viewport for col in COLLS}
    for col in COLLS :
        col.hide_viewport = False

    for col in bpy.data.collections :
        col.hide_viewport = False

    WM = bpy.context.window_manager

    # Duplicate Area3D to new window :
    MainWindow = WM.windows[0]
    LayoutScreen = bpy.data.screens["Layout"]
    LayoutArea3D = [area for area in LayoutScreen.areas if area.type == "VIEW_3D"][0]

    Override = {"window": MainWindow, "screen": LayoutScreen, "area": LayoutArea3D}
    bpy.ops.screen.area_dupli(Override, "INVOKE_DEFAULT")

    # Get MultiView (Window, Screen, Area3D, Space3D, Region3D) and set prefferences :
    MultiView_Window = WM.windows[-1]
    MultiView_Screen = MultiView_Window.screen

    MultiView_Area3D = [
        area for area in MultiView_Screen.areas if area.type == "VIEW_3D"
    ][0]
    MultiView_Space3D = [
        space for space in MultiView_Area3D.spaces if space.type == "VIEW_3D"
    ][0]
    MultiView_Region3D = [
        reg for reg in MultiView_Area3D.regions if reg.type == "WINDOW"
    ][0]

    MultiView_Area3D.type = (
        "CONSOLE"  # change area type for update : bug dont respond to spliting
    )

    # 1rst Step : Vertical Split .

    Override = {
        "window": MultiView_Window,
        "screen": MultiView_Screen,
        "area": MultiView_Area3D,
        "space_data": MultiView_Space3D,
        "region": MultiView_Region3D,
    }

    bpy.ops.screen.area_split(Override, direction="VERTICAL", factor=1 / 5)
    MultiView_Screen.areas[0].type = "OUTLINER"
    MultiView_Screen.areas[1].type = "OUTLINER"

    # 2nd Step : Horizontal Split .
    Active_Area = MultiView_Screen.areas[0]
    Active_Space = [space for space in Active_Area.spaces if space.type == "VIEW_3D"][0]
    Active_Region = [reg for reg in Active_Area.regions if reg.type == "WINDOW"][0]
    Override = {
        "window": MultiView_Window,
        "screen": MultiView_Screen,
        "area": Active_Area,
        "space_data": Active_Space,
        "region": Active_Region,
    }

    bpy.ops.screen.area_split(Override, direction="HORIZONTAL", factor=1 / 2)
    MultiView_Screen.areas[0].type = "VIEW_3D"
    MultiView_Screen.areas[1].type = "VIEW_3D"
    MultiView_Screen.areas[2].type = "VIEW_3D"

    # 3rd Step : Vertical Split .
    Active_Area = MultiView_Screen.areas[0]
    Active_Space = [space for space in Active_Area.spaces if space.type == "VIEW_3D"][0]
    Active_Region = [reg for reg in Active_Area.regions if reg.type == "WINDOW"][0]
    Override = {
        "window": MultiView_Window,
        "screen": MultiView_Screen,
        "area": Active_Area,
        "space_data": Active_Space,
        "region": Active_Region,
    }

    bpy.ops.screen.area_split(Override, direction="VERTICAL", factor=1 / 2)
    MultiView_Screen.areas[0].type = "OUTLINER"
    MultiView_Screen.areas[1].type = "OUTLINER"
    MultiView_Screen.areas[2].type = "OUTLINER"
    MultiView_Screen.areas[3].type = "OUTLINER"

    # 4th Step : Vertical Split .
    Active_Area = MultiView_Screen.areas[2]
    Active_Space = [space for space in Active_Area.spaces if space.type == "VIEW_3D"][0]
    Active_Region = [reg for reg in Active_Area.regions if reg.type == "WINDOW"][0]
    Override = {
        "window": MultiView_Window,
        "screen": MultiView_Screen,
        "area": Active_Area,
        "space_data": Active_Space,
        "region": Active_Region,
    }

    bpy.ops.screen.area_split(Override, direction="VERTICAL", factor=1 / 2)
    MultiView_Screen.areas[0].type = "VIEW_3D"
    MultiView_Screen.areas[1].type = "VIEW_3D"
    MultiView_Screen.areas[2].type = "VIEW_3D"
    MultiView_Screen.areas[3].type = "VIEW_3D"
    MultiView_Screen.areas[4].type = "VIEW_3D"

    # 4th Step : Horizontal Split .
    Active_Area = MultiView_Screen.areas[1]
    Active_Space = [space for space in Active_Area.spaces if space.type == "VIEW_3D"][0]
    Active_Region = [reg for reg in Active_Area.regions if reg.type == "WINDOW"][0]
    Override = {
        "window": MultiView_Window,
        "screen": MultiView_Screen,
        "area": Active_Area,
        "space_data": Active_Space,
        "region": Active_Region,
    }

    bpy.ops.screen.area_split(Override, direction="HORIZONTAL", factor=1 / 2)

    MultiView_Screen.areas[1].type = "OUTLINER"
    MultiView_Screen.areas[5].type = "PROPERTIES"

    # Set MultiView Areas 3D prefferences :
    ##### Hide local collections :
    collNames = [col.name for col in bpy.context.scene.collection.children if not ('SLICES' in col.name or 'SLICES_POINTERS' in col.name or "GUIDE Components" in col.name )]
    
    for i, MultiView_Area3D in enumerate(MultiView_Screen.areas):

        if MultiView_Area3D.type == "VIEW_3D":
            MultiView_Space3D = [
                space for space in MultiView_Area3D.spaces if space.type == "VIEW_3D"
            ][0]

            Override = {
                "window": MultiView_Window,
                "screen": MultiView_Screen,
                "area": MultiView_Area3D,
                "space_data": MultiView_Space3D,
            }

            bpy.ops.wm.tool_set_by_id(Override, name="builtin.move")
            MultiView_Space3D.use_local_collections = True
            if not i == 4 : 
                for collName in collNames :
                    index = getLocalCollIndex(collName)
                    bpy.ops.object.hide_collection(Override, collection_index=index,toggle=True)


            MultiView_Space3D.overlay.show_text = True
            MultiView_Space3D.show_region_ui = False
            MultiView_Space3D.show_region_toolbar = True
            MultiView_Space3D.region_3d.view_perspective = "ORTHO"
            MultiView_Space3D.show_gizmo_navigate = False
            MultiView_Space3D.show_region_tool_header = False
            MultiView_Space3D.overlay.show_floor = False
            MultiView_Space3D.overlay.show_ortho_grid = False
            MultiView_Space3D.overlay.show_relationship_lines = False
            MultiView_Space3D.overlay.show_extras = True
            MultiView_Space3D.overlay.show_bones = False
            MultiView_Space3D.overlay.show_motion_paths = False

            MultiView_Space3D.shading.type = "SOLID"
            MultiView_Space3D.shading.light = "STUDIO"
            MultiView_Space3D.shading.studio_light = "outdoor.sl"
            MultiView_Space3D.shading.color_type = "TEXTURE"
            MultiView_Space3D.shading.background_type = "VIEWPORT"
            MultiView_Space3D.shading.background_color = [0.0, 0.0, 0.0]#[0.7, 0.7, 0.7]

            MultiView_Space3D.shading.type = "MATERIAL"
            # 'Material' Shading Light method :
            MultiView_Space3D.shading.use_scene_lights = True
            MultiView_Space3D.shading.use_scene_world = False

            # 'RENDERED' Shading Light method :
            MultiView_Space3D.shading.use_scene_lights_render = False
            MultiView_Space3D.shading.use_scene_world_render = True

            MultiView_Space3D.shading.studio_light = "forest.exr"
            MultiView_Space3D.shading.studiolight_rotate_z = 0
            MultiView_Space3D.shading.studiolight_intensity = 1.5
            MultiView_Space3D.shading.studiolight_background_alpha = 0.0
            MultiView_Space3D.shading.studiolight_background_blur = 0.0

            MultiView_Space3D.shading.render_pass = "COMBINED"

            MultiView_Space3D.show_region_header = False

    OUTLINER = TopLeft = MultiView_Screen.areas[1]
    PROPERTIES = DownLeft = MultiView_Screen.areas[5]
    AXIAL = TopMiddle = MultiView_Screen.areas[3]
    CORONAL = TopRight = MultiView_Screen.areas[0]
    SAGITAL = DownRight = MultiView_Screen.areas[2]
    VIEW_3D = DownMiddle = MultiView_Screen.areas[4]

    for col in COLLS :
        col.hide_viewport = collectionState[col]

    #    TopMiddle.header_text_set("AXIAL")
    #    TopRight.header_text_set("CORONAL")
    #    DownRight.header_text_set("SAGITAL")
    #    DownMiddle.header_text_set("3D VIEW")

    return MultiView_Window, OUTLINER, PROPERTIES, AXIAL, CORONAL, SAGITAL, VIEW_3D


##############################################
# Vertex Paint Cutter :


def VertexPaintCut(mode):

    #######################################################################################

    # start = time.perf_counter()
    ActiveObj = bpy.context.active_object
    paint_color = bpy.data.brushes["Draw"].color
    dict_paint_color = {
        "r_channel": bpy.data.brushes["Draw"].color.r,
        "g_channel": bpy.data.brushes["Draw"].color.g,
        "b_channel": bpy.data.brushes["Draw"].color.b,
    }
    r_channel = bpy.data.brushes["Draw"].color.r
    g_channel = bpy.data.brushes["Draw"].color.g
    b_channel = bpy.data.brushes["Draw"].color.b

    list_colored_verts_indices = []
    list_colored_verts_colors = []
    dict_vid_vcolor = {}

    # get ActiveObj, hide everything but ActiveObj :
    bpy.ops.object.mode_set(mode="OBJECT")
    mesh = ActiveObj.data
    bpy.ops.object.select_all(action="DESELECT")
    ActiveObj.select_set(True)

    if len(bpy.context.visible_objects) > 1:
        bpy.ops.object.hide_view_set(unselected=True)

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")

    # Make dictionary : key= vertex index , value = vertex color(RGB)
    for polygon in mesh.polygons:

        for v_poly_index, v_global_index in enumerate(polygon.vertices):

            col_index = polygon.loop_indices[v_poly_index]
            v_color = mesh.vertex_colors.active.data[col_index].color[:]
            dict_vid_vcolor[v_global_index] = v_color

    # calculate averrage color :
    paint_color = tuple(bpy.data.brushes["Draw"].color)
    white_color = (1, 1, 1)
    color_offset = (1 - paint_color[0], 1 - paint_color[1], 1 - paint_color[2])
    # distance = sqrt((1-paint_color[0])**2+(1-paint_color[1])**2+pow(1-paint_color[2])**2)
    factor = 0.5
    average_color = (
        paint_color[0] + factor * color_offset[0],
        paint_color[1] + factor * color_offset[1],
        paint_color[2] + factor * color_offset[2],
    )

    # Make list : collect indices of colored vertices
    for key, value in dict_vid_vcolor.items():
        if paint_color <= value[0:3] <= average_color:
            list_colored_verts_indices.append(key)
            list_colored_verts_colors.append(value)

    # select colored verts :
    for i in list_colored_verts_indices:
        mesh.vertices[i].select = True

    # remove old vertex_groups and make new one :
    bpy.ops.object.mode_set(mode="EDIT")

    for vg in ActiveObj.vertex_groups:
        if "BDENTAL_4D_PaintCutter_" in vg.name:
            ActiveObj.vertex_groups.remove(vg)

    Area_vg = ActiveObj.vertex_groups.new(name="BDENTAL_4D_PaintCutter_Area_vg")
    ActiveObj.vertex_groups.active_index = Area_vg.index
    bpy.ops.object.vertex_group_assign()
    bpy.ops.mesh.region_to_loop()
    Border_vg = ActiveObj.vertex_groups.new(name="BDENTAL_4D_PaintCutter_Border_vg")
    bpy.ops.object.vertex_group_assign()

    Addon_Enable(AddonName="mesh_looptools", Enable=True)
    bpy.ops.mesh.looptools_relax(
        input="selected", interpolation="cubic", iterations="5", regular=True
    )

    if mode == "Cut":
        bpy.ops.mesh.loop_to_region()
        bpy.ops.bdental4d.separate_objects(SeparateMode="Selection")

    if mode == "Make Copy (Shell)":
        bpy.ops.mesh.loop_to_region()
        # duplicate selected verts, separate and make splint shell
        bpy.ops.mesh.duplicate_move()
        bpy.ops.mesh.separate(type="SELECTED")
        bpy.ops.object.mode_set(mode="OBJECT")
        shell = bpy.context.selected_objects[1]
        bpy.ops.object.select_all(action="DESELECT")
        shell.select_set(True)
        bpy.context.view_layer.objects.active = shell

        shell.name = "Shell"
        # Add color material :
        mat = bpy.data.materials.get(
            "BDENTAL_4D_PaintCut_mat"
        ) or bpy.data.materials.new("BDENTAL_4D_PaintCut_mat")
        mat.diffuse_color = [paint_color[0], paint_color[1], paint_color[2], 1]
        shell.active_material = mat

    if mode == "Remove Painted":
        bpy.ops.mesh.loop_to_region()
        bpy.ops.mesh.delete(type="FACE")
        bpy.ops.object.mode_set(mode="OBJECT")

    if mode == "Keep Painted":
        bpy.ops.mesh.loop_to_region()
        bpy.ops.mesh.select_all(action="INVERT")
        bpy.ops.mesh.delete(type="VERT")
        bpy.ops.object.mode_set(mode="OBJECT")


def CursorToVoxelPoint(Preffix, CursorMove=False):

    VoxelPointCo = 0
    CTVolume = bpy.data.objects.get(f"{Preffix}_CTVolume")
    TransformMatrix = CTVolume.matrix_world
    BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props
    DcmInfoDict = eval(BDENTAL_4D_Props.DcmInfo)
    DcmInfo = DcmInfoDict[Preffix]
    ImageData = bpy.path.abspath(DcmInfo["Nrrd255Path"])
    Treshold = BDENTAL_4D_Props.Treshold
    Wmin, Wmax = DcmInfo["Wmin"], DcmInfo["Wmax"]

    Cursor = bpy.context.scene.cursor
    CursorInitMtx = Cursor.matrix.copy()

    # Get ImageData Infos :
    Image3D_255 = sitk.ReadImage(ImageData)
    Sp = Spacing = Image3D_255.GetSpacing()
    Sz = Size = Image3D_255.GetSize()
    Ortho_Origin = -0.5 * np.array(Sp) * (np.array(Sz) - np.array((1, 1, 1)))
    Image3D_255.SetOrigin(Ortho_Origin)
    Image3D_255.SetDirection(np.identity(3).flatten())

    # Cursor shift :
    Cursor_Z = Vector((CursorInitMtx[0][2], CursorInitMtx[1][2], CursorInitMtx[2][2]))
    CT = CursorTrans = -1 * (Sz[2] - 1) * Sp[2] * Cursor_Z
    CursorTransMatrix = mathutils.Matrix(
        (
            (1.0, 0.0, 0.0, CT[0]),
            (0.0, 1.0, 0.0, CT[1]),
            (0.0, 0.0, 1.0, CT[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    )

    # Output Parameters :
    Out_Origin = [Ortho_Origin[0], Ortho_Origin[1], 0]
    Out_Direction = Vector(np.identity(3).flatten())
    Out_Size = Sz
    Out_Spacing = Sp

    # Get Plane Orientation and location :
    MyMatrix = TransformMatrix.inverted() @ CursorTransMatrix @ CursorInitMtx
    Rot = MyMatrix.to_euler()
    Rvec = (Rot.x, Rot.y, Rot.z)
    Tvec = MyMatrix.translation

    # Euler3DTransform :
    Euler3D = sitk.Euler3DTransform()
    Euler3D.SetCenter((0, 0, 0))
    Euler3D.SetRotation(Rvec[0], Rvec[1], Rvec[2])
    Euler3D.SetTranslation(Tvec)
    Euler3D.ComputeZYXOn()

    #########################################

    Image3D = sitk.Resample(
        Image3D_255,
        Out_Size,
        Euler3D,
        sitk.sitkLinear,
        Out_Origin,
        Out_Spacing,
        Out_Direction,
        0,
    )

    #  # Write Image :
    # Array = sitk.GetArrayFromImage(Image3D[:,:,Sz[2]-1])#Sz[2]-1
    # Flipped_Array = np.flipud(Array.reshape(Array.shape[0], Array.shape[1]))
    # cv2.imwrite(ImagePath, Flipped_Array)

    ImgArray = sitk.GetArrayFromImage(Image3D)
    Treshold255 = int(((Treshold - Wmin) / (Wmax - Wmin)) * 255)

    RayPixels = ImgArray[:, int(Sz[1] / 2), int(Sz[0] / 2)]
    ReversedRayPixels = list(reversed(list(RayPixels)))

    for i, P in enumerate(ReversedRayPixels):
        if P >= Treshold255:
            VoxelPointCo = Cursor.location - i * Sp[2] * Cursor_Z
            break

    if CursorMove and VoxelPointCo:
        bpy.context.scene.cursor.location = VoxelPointCo
    #############################################

    return VoxelPointCo


def Metaball_Splint(shell, thikness):
    #############################################################
    # Add Metaballs :

    radius = thikness * 5 / 8
    bpy.ops.object.select_all(action="DESELECT")
    shell.select_set(True)
    bpy.context.view_layer.objects.active = shell

    vcords = [shell.matrix_world @ v.co for v in shell.data.vertices]
    mball_elements_cords = [vco - vcords[0] for vco in vcords[1:]]

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")

    bpy.ops.object.metaball_add(
        type="BALL", radius=radius, enter_editmode=False, location=vcords[0]
    )

    mball_obj = bpy.context.view_layer.objects.active

    mball = mball_obj.data
    mball.resolution = 0.6
    bpy.context.object.data.update_method = "FAST"

    for i in range(len(mball_elements_cords)):
        element = mball.elements.new()
        element.co = mball_elements_cords[i]
        element.radius = radius * 2

    bpy.ops.object.convert(target="MESH")

    splint = bpy.context.view_layer.objects.active
    splint.name = "BDENTAL4D_Splint"
    splint_mesh = splint.data
    splint_mesh.name = "BDENTAL4D_Splint_mesh"

    mat = bpy.data.materials.get("BDENTAL4D_splint_mat") or bpy.data.materials.new(
        "BDENTAL4D_splint_mat"
    )
    mat.diffuse_color = [0.0, 0.6, 0.8, 1.0]
    splint.active_material = mat
    bpy.ops.object.select_all(action="DESELECT")

    return splint


######################################################################################
# Align Utils
######################################################################################


def AddRefPoint(name, color, CollName=None):

    loc = bpy.context.scene.cursor.location
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.2, location=loc)
    RefP = bpy.context.object
    RefP.name = name
    RefP.data.name = name + "_mesh"
    if CollName:
        MoveToCollection(RefP, CollName)
    if name.startswith("B"):
        matName = "TargetRefMat"
    if name.startswith("M"):
        matName = "SourceRefMat"

    mat = bpy.data.materials.get(matName) or bpy.data.materials.new(matName)
    mat.use_nodes = False
    mat.diffuse_color = color
    RefP.active_material = mat
    RefP.show_name = True
    return RefP


def RefPointsToTransformMatrix(TargetRefPoints, SourceRefPoints):
    # TransformMatrix = Matrix()  # identity Matrix (4x4)

    # make 2 arrays of coordinates :
    TargetArray = np.array(
        [obj.location for obj in TargetRefPoints], dtype=np.float64
    ).T
    SourceArray = np.array(
        [obj.location for obj in SourceRefPoints], dtype=np.float64
    ).T

    # Calculate centers of Target and Source RefPoints :
    TargetCenter, SourceCenter = np.mean(TargetArray, axis=1), np.mean(
        SourceArray, axis=1
    )

    # Calculate Translation :
    ###################################

    # TransMatrix_1 : Matrix(4x4) will translate center of SourceRefPoints...
    # to origine (0,0,0) location.
    TransMatrix_1 = Matrix.Translation(Vector(-SourceCenter))

    # TransMatrix_2 : Matrix(4x4) will translate center of SourceRefPoints...
    #  to the center of TargetRefPoints location.
    TransMatrix_2 = Matrix.Translation(Vector(TargetCenter))

    # Calculate Rotation :
    ###################################

    # Home Arrays will get the Centered Target and Source RefPoints around origin (0,0,0).
    HomeTargetArray, HomeSourceArray = (
        TargetArray - TargetCenter.reshape(3, 1),
        SourceArray - SourceCenter.reshape(3, 1),
    )
    # Rigid transformation via SVD of covariance matrix :
    U, S, Vt = np.linalg.svd(np.dot(HomeTargetArray, HomeSourceArray.T))

    # rotation matrix from SVD orthonormal bases and check,
    # if it is a Reflection matrix :
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0.0:
        Vt[2, :] *= -1
        R = np.dot(U, Vt)
        print(" Reflection matrix fixed ")

    RotationMatrix = Matrix(R).to_4x4()
    TransformMatrix = TransMatrix_2 @ RotationMatrix @ TransMatrix_1

    return TransformMatrix


def KdIcpPairs(SourceVcoList, TargetVcolist, VertsLimite=5000):
    start = Tcounter()
    # print("KD processing start...")
    SourceKdList, TargetKdList, DistList, SourceIndexList, TargetIndexList = (
        [],
        [],
        [],
        [],
        [],
    )
    size = len(TargetVcolist)
    kd = kdtree.KDTree(size)

    for i, Vco in enumerate(TargetVcolist):
        kd.insert(Vco, i)

    kd.balance()

    n = len(SourceVcoList)
    if n > VertsLimite:
        step = ceil(n / VertsLimite)
        SourceVcoList = SourceVcoList[::step]

    for SourceIndex, Sco in enumerate(SourceVcoList):

        Tco, TargetIndex, dist = kd.find(Sco)
        if Tco:
            if not TargetIndex in TargetIndexList:
                TargetIndexList.append(TargetIndex)
                SourceIndexList.append(SourceIndex)
                TargetKdList.append(Tco)
                SourceKdList.append(Sco)
                DistList.append(dist)
    finish = Tcounter()
    # print(f"KD total iterations : {len(SourceVcoList)}")
    # print(f"KD Index List : {len(IndexList)}")

    # print(f"KD finshed in {finish-start} secondes")
    return SourceKdList, TargetKdList, DistList, SourceIndexList, TargetIndexList


def KdRadiusVerts(obj, RefCo, radius):

    RadiusVertsIds = []
    RadiusVertsCo = []
    RadiusVertsDistance = []
    verts = obj.data.vertices
    Vcolist = [obj.matrix_world @ v.co for v in verts]
    size = len(Vcolist)
    kd = kdtree.KDTree(size)

    for i, Vco in enumerate(Vcolist):
        kd.insert(Vco, i)

    kd.balance()

    for (co, index, dist) in kd.find_range(RefCo, radius):

        RadiusVertsIds.append(index)
        RadiusVertsCo.append(co)
        RadiusVertsDistance.append(dist)

    return RadiusVertsIds, RadiusVertsCo, RadiusVertsDistance


def VidDictFromPoints(TargetRefPoints, SourceRefPoints, TargetObj, SourceObj, radius):
    IcpVidDict = {TargetObj: [], SourceObj: []}

    for obj in [TargetObj, SourceObj]:
        if obj == TargetObj:
            for RefTargetP in TargetRefPoints:
                RefCo = RefTargetP.location
                RadiusVertsIds, RadiusVertsCo, RadiusVertsDistance = KdRadiusVerts(
                    TargetObj, RefCo, radius
                )
                IcpVidDict[TargetObj].extend(RadiusVertsIds)
                for idx in RadiusVertsIds:
                    obj.data.vertices[idx].select = True
        if obj == SourceObj:
            for RefSourceP in SourceRefPoints:
                RefCo = RefSourceP.location
                RadiusVertsIds, RadiusVertsCo, RadiusVertsDistance = KdRadiusVerts(
                    SourceObj, RefCo, radius
                )
                IcpVidDict[SourceObj].extend(RadiusVertsIds)
                for idx in RadiusVertsIds:
                    obj.data.vertices[idx].select = True

    bpy.ops.object.select_all(action="DESELECT")
    for obj in [TargetObj, SourceObj]:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = TargetObj

    return IcpVidDict


def KdIcpPairsToTransformMatrix(TargetKdList, SourceKdList):
    # make 2 arrays of coordinates :
    TargetArray = np.array(TargetKdList, dtype=np.float64).T
    SourceArray = np.array(SourceKdList, dtype=np.float64).T

    # Calculate centers of Target and Source RefPoints :
    TargetCenter, SourceCenter = np.mean(TargetArray, axis=1), np.mean(
        SourceArray, axis=1
    )

    # Calculate Translation :
    ###################################

    # TransMatrix_1 : Matrix(4x4) will translate center of SourceRefPoints...
    # to origine (0,0,0) location.
    TransMatrix_1 = Matrix.Translation(Vector(-SourceCenter))

    # TransMatrix_2 : Matrix(4x4) will translate center of SourceRefPoints...
    #  to the center of TargetRefPoints location.
    TransMatrix_2 = Matrix.Translation(Vector(TargetCenter))

    # Calculate Rotation :
    ###################################

    # Home Arrays will get the Centered Target and Source RefPoints around origin (0,0,0).
    HomeTargetArray, HomeSourceArray = (
        TargetArray - TargetCenter.reshape(3, 1),
        SourceArray - SourceCenter.reshape(3, 1),
    )
    # Rigid transformation via SVD of covariance matrix :
    U, S, Vt = np.linalg.svd(np.dot(HomeTargetArray, HomeSourceArray.T))

    # rotation matrix from SVD orthonormal bases :
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0.0:
        Vt[2, :] *= -1
        R = np.dot(U, Vt)
        print(" Reflection fixed ")

    RotationMatrix = Matrix(R).to_4x4()
    TransformMatrix = TransMatrix_2 @ RotationMatrix @ TransMatrix_1

    return TransformMatrix


def AddVoxelPoint(
    Name="Voxel Anatomical Point",
    Color=(1.0, 0.0, 0.0, 1.0),
    Location=(0, 0, 0),
    Radius=1.2,
):
    Active_Obj = bpy.context.view_layer.objects.active
    bpy.ops.mesh.primitive_uv_sphere_add(radius=Radius, location=Location)
    Sphere = bpy.context.object
    Sphere.name = Name
    Sphere.data.name = Name

    MoveToCollection(Sphere, "VOXELS Points")

    matName = f"VOXEL_Points_Mat"
    mat = bpy.data.materials.get(matName) or bpy.data.materials.new(matName)
    mat.diffuse_color = Color
    mat.use_nodes = False
    Sphere.active_material = mat
    Sphere.show_name = True
    bpy.ops.object.select_all(action="DESELECT")
    Active_Obj.select_set(True)
    bpy.context.view_layer.objects.active = Active_Obj


def CursorToVoxelPoint(Preffix, CursorMove=False):

    VoxelPointCo = 0

    BDENTAL_4D_Props = bpy.context.scene.BDENTAL_4D_Props
    DcmInfoDict = eval(BDENTAL_4D_Props.DcmInfo)
    DcmInfo = DcmInfoDict[Preffix]
    ImageData = bpy.path.abspath(DcmInfo["Nrrd255Path"])
    Treshold = BDENTAL_4D_Props.Treshold
    Wmin, Wmax = DcmInfo["Wmin"], DcmInfo["Wmax"]
    TransformMatrix = DcmInfo["TransformMatrix"]
    VtkTransform_4x4 = DcmInfo["VtkTransform_4x4"]

    Cursor = bpy.context.scene.cursor
    CursorInitMtx = Cursor.matrix.copy()

    # Get ImageData Infos :
    Image3D_255 = sitk.ReadImage(ImageData)
    Sp = Spacing = Image3D_255.GetSpacing()
    Sz = Size = Image3D_255.GetSize()
    Ortho_Origin = -0.5 * np.array(Sp) * (np.array(Sz) - np.array((1, 1, 1)))
    Image3D_255.SetOrigin(Ortho_Origin)
    Image3D_255.SetDirection(np.identity(3).flatten())

    # Cursor shift :
    Cursor_Z = Vector((CursorInitMtx[0][2], CursorInitMtx[1][2], CursorInitMtx[2][2]))
    CT = CursorTrans = -1 * (Sz[2] - 1) * Sp[2] * Cursor_Z
    CursorTransMatrix = mathutils.Matrix(
        (
            (1.0, 0.0, 0.0, CT[0]),
            (0.0, 1.0, 0.0, CT[1]),
            (0.0, 0.0, 1.0, CT[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    )

    # Output Parameters :
    Out_Origin = [Ortho_Origin[0], Ortho_Origin[1], 0]
    Out_Direction = Vector(np.identity(3).flatten())
    Out_Size = Sz
    Out_Spacing = Sp

    # Get Plane Orientation and location :
    Matrix = TransformMatrix.inverted() @ CursorTransMatrix @ CursorInitMtx
    Rot = Matrix.to_euler()
    Rvec = (Rot.x, Rot.y, Rot.z)
    Tvec = Matrix.translation

    # Euler3DTransform :
    Euler3D = sitk.Euler3DTransform()
    Euler3D.SetCenter((0, 0, 0))
    Euler3D.SetRotation(Rvec[0], Rvec[1], Rvec[2])
    Euler3D.SetTranslation(Tvec)
    Euler3D.ComputeZYXOn()

    #########################################

    Image3D = sitk.Resample(
        Image3D_255,
        Out_Size,
        Euler3D,
        sitk.sitkLinear,
        Out_Origin,
        Out_Spacing,
        Out_Direction,
        0,
    )

    #  # Write Image :
    # Array = sitk.GetArrayFromImage(Image3D[:,:,Sz[2]-1])#Sz[2]-1
    # Flipped_Array = np.flipud(Array.reshape(Array.shape[0], Array.shape[1]))
    # cv2.imwrite(ImagePath, Flipped_Array)

    ImgArray = sitk.GetArrayFromImage(Image3D)
    Treshold255 = int(((Treshold - Wmin) / (Wmax - Wmin)) * 255)

    RayPixels = ImgArray[:, int(Sz[1] / 2), int(Sz[0] / 2)]
    ReversedRayPixels = list(reversed(list(RayPixels)))

    for i, P in enumerate(ReversedRayPixels):
        if P >= Treshold255:
            VoxelPointCo = Cursor.location - i * Sp[2] * Cursor_Z
            break

    if CursorMove and VoxelPointCo:
        bpy.context.scene.cursor.location = VoxelPointCo
    #############################################

    return VoxelPointCo


#################################################################################
# JTrack utils
#################################################################################
def JTrackRepportPlot(ImgFolder):

    IP_CP_X, IP_CP_Y, IP_CP_Z = bpy.data.objects["IP_Central"].location
    LC_CP_X, LC_CP_Y, LC_CP_Z = bpy.data.objects["LC_Central"].location
    RC_CP_X, RC_CP_Y, RC_CP_Z = bpy.data.objects["RC_Central"].location

    IP_LL_X, IP_LL_Y, IP_LL_Z = bpy.data.objects["IP_Lateral-Left"].location - Vector(
        (IP_CP_X, IP_CP_Y, IP_CP_Z)
    )
    IP_LR_X, IP_LR_Y, IP_LR_Z = bpy.data.objects["IP_Lateral-Right"].location - Vector(
        (IP_CP_X, IP_CP_Y, IP_CP_Z)
    )
    IP_MO_X, IP_MO_Y, IP_MO_Z = bpy.data.objects["IP_Max-Open"].location - Vector(
        (IP_CP_X, IP_CP_Y, IP_CP_Z)
    )
    IP_PP_X, IP_PP_Y, IP_PP_Z = bpy.data.objects["IP-Protrusion"].location - Vector(
        (IP_CP_X, IP_CP_Y, IP_CP_Z)
    )

    LC_LL_X, LC_LL_Y, LC_LL_Z = bpy.data.objects["LC_Lateral-Left"].location - Vector(
        (LC_CP_X, LC_CP_Y, LC_CP_Z)
    )
    LC_LR_X, LC_LR_Y, LC_LR_Z = bpy.data.objects["LC_Lateral-Right"].location - Vector(
        (LC_CP_X, LC_CP_Y, LC_CP_Z)
    )
    LC_MO_X, LC_MO_Y, LC_MO_Z = bpy.data.objects["LC_Max-Open"].location - Vector(
        (LC_CP_X, LC_CP_Y, LC_CP_Z)
    )
    LC_PP_X, LC_PP_Y, LC_PP_Z = bpy.data.objects["LC-Protrusion"].location - Vector(
        (LC_CP_X, LC_CP_Y, LC_CP_Z)
    )

    RC_LL_X, RC_LL_Y, RC_LL_Z = bpy.data.objects["RC_Lateral-Left"].location - Vector(
        (RC_CP_X, RC_CP_Y, RC_CP_Z)
    )
    RC_LR_X, RC_LR_Y, RC_LR_Z = bpy.data.objects["RC_Lateral-Right"].location - Vector(
        (RC_CP_X, RC_CP_Y, RC_CP_Z)
    )
    RC_MO_X, RC_MO_Y, RC_MO_Z = bpy.data.objects["RC_Max-Open"].location - Vector(
        (RC_CP_X, RC_CP_Y, RC_CP_Z)
    )
    RC_PP_X, RC_PP_Y, RC_PP_Z = bpy.data.objects["RC-Protrusion"].location - Vector(
        (RC_CP_X, RC_CP_Y, RC_CP_Z)
    )

    LC_Array = GetEmptyMovementsArray(EmptyName="Left Condyle")
    LC_X_Array, LC_Y_Array, LC_Z_Array = (
        LC_Array[:, 0] - LC_CP_X,
        LC_Array[:, 1] - LC_CP_Y,
        LC_Array[:, 2] - LC_CP_Z,
    )

    RC_Array = GetEmptyMovementsArray(EmptyName="Right Condyle")
    RC_X_Array, RC_Y_Array, RC_Z_Array = (
        RC_Array[:, 0] - RC_CP_X,
        RC_Array[:, 1] - RC_CP_Y,
        RC_Array[:, 2] - RC_CP_Z,
    )

    IP_Array = GetEmptyMovementsArray(EmptyName="Incisal")
    IP_X_Array, IP_Y_Array, IP_Z_Array = (
        IP_Array[:, 0] - IP_CP_X,
        IP_Array[:, 1] - IP_CP_Y,
        IP_Array[:, 2] - IP_CP_Z,
    )

    fig = plt.figure(figsize=(11.69, 18))
    fig.patch.set_facecolor("silver")
    spec = gridspec.GridSpec(
        ncols=4,
        nrows=6,
        figure=fig,
        left=0.07,
        bottom=0.13,
        right=0.95,
        top=0.97,
        wspace=0.9,
        hspace=1.5,
    )
    infospec = gridspec.GridSpec(
        ncols=4,
        nrows=12,
        figure=fig,
        left=0.07,
        bottom=0.1,
        right=0.95,
        top=1,
        wspace=0.9,
        hspace=1.5,
    )
    headimgalpha = 0.5
    ###################################################################################
    # INCISAL LINES
    ###################################################################################
    IP_CP_X, IP_CP_Y, IP_CP_Z = 0, 0, 0

    LineOpenX = [IP_CP_X, IP_MO_X]
    LineOpenY = [IP_CP_Y, IP_MO_Y]
    LineOpenZ = [IP_CP_Z, IP_MO_Z]

    LineProtrusionX = [IP_CP_X, IP_PP_X]
    LineProtrusionY = [IP_CP_Y, IP_PP_Y]
    LineProtrusionZ = [IP_CP_Z, IP_PP_Z]

    LineLatRightX = [IP_CP_X, IP_LR_X]
    LineLatRightY = [IP_CP_Y, IP_LR_Y]
    LineLatRightZ = [IP_CP_Z, IP_LR_Z]

    LineLatLeftX = [IP_CP_X, IP_LL_X]
    LineLatLeftY = [IP_CP_Y, IP_LL_Y]
    LineLatLeftZ = [IP_CP_Z, IP_LL_Z]

    LineMaxOpenY = [IP_MO_Y, IP_MO_Y]
    LineMaxOpenZ = [IP_CP_Z, IP_MO_Z]

    # FRONTAL PLANE###########################################################################
    ax0 = fig.add_subplot(spec[0:2, 0:2])
    ax0.patch.set_facecolor("whitesmoke")
    ax0.set_title("Coronal Plane (incisal point)")
    ax0.set(xlabel="X axis, mm", ylabel="Z axis, mm")
    ax0.axis('equal')
    ax0.xaxis.set_major_locator(MultipleLocator(5))
    ax0.yaxis.set_major_locator(MultipleLocator(5))
    ax0.xaxis.set_minor_locator(MultipleLocator(1))
    ax0.yaxis.set_minor_locator(MultipleLocator(1))
    ax0.grid(which="minor", color="#e4e4e4", linestyle="--")
    
    ax0.grid(which="major", color="#CCCCCC", linestyle="--")
    ax0.grid(True)
    ax0.plot(IP_X_Array, IP_Z_Array, color="dimgray", linewidth=0.5)
    ax0.axhline(y=IP_CP_Z, color="black", linestyle="--", linewidth=2)
    ax0.axvline(x=IP_CP_X, color="black", linestyle="--", linewidth=2)
    ax0.plot(
        LineOpenX,
        LineOpenZ,
        color="red",
        linewidth=2,
        marker="o",
        mfc="black",
        mec="black",
        ms=6,
        label="Open",
    )
    ax0.plot(
        LineProtrusionX,
        LineProtrusionZ,
        color="blue",
        linewidth=2,
        marker="o",
        mfc="black",
        mec="black",
        ms=6,
        label="Protrusion",
    )
    ax0.plot(
        LineLatRightX,
        LineLatRightZ,
        color="orange",
        linewidth=2,
        marker="o",
        mfc="black",
        mec="black",
        ms=6,
        label="Laterotrusion Right",
    )
    ax0.plot(
        LineLatLeftX,
        LineLatLeftZ,
        color="magenta",
        linewidth=2,
        marker="o",
        mfc="black",
        mec="black",
        ms=6,
        label="Laterotrusion Left",
    )

    plt.legend(markerscale=0)

    ax = fig.add_subplot(spec[1, 1])
    image = plt.imread(join(ImgFolder, "incfront.png"))
    ax.imshow(image, alpha=headimgalpha)
    ax.axis("off")

    # ANGLES
    # RIGHT LATEROTRUSION
    center = (IP_CP_X, IP_CP_Z)
    p1 = np.array([(IP_CP_X, IP_CP_Z), (IP_LR_X, IP_CP_Z)])
    p2 = np.array([(IP_CP_X, IP_CP_Z), (IP_LR_X, IP_LR_Z)])
    
    a=np.array([IP_LR_X, IP_CP_Z])
    b=np.array([IP_CP_X, IP_CP_Z])
    c=np.array([IP_LR_X, IP_LR_Z])
    ANGLE_LR = round(np.degrees(getAngle(a, b, c)), 2)
    if IP_LR_Z>IP_CP_Z:
        am0 = AngleAnnotation(center, p2[1], p1[1], ax=ax0, size=100, text=ANGLE_LR, linewidth=3, zorder=10)
    else:
        am0 = AngleAnnotation(center, p1[1], p2[1], ax=ax0, size=100, text=ANGLE_LR, linewidth=3, zorder=10)

    # LEFT LATEROTRUSION
    center = (IP_CP_X, IP_CP_Z)
    p1 = np.array([(IP_CP_X, IP_CP_Z), (IP_LL_X, IP_CP_Z)])
    p2 = np.array([(IP_CP_X, IP_CP_Z), (IP_LL_X, IP_LL_Z)])

    a=np.array([IP_LL_X, IP_CP_Z])
    b=np.array([IP_CP_X, IP_CP_Z])
    c=np.array([IP_LL_X, IP_LL_Z])
    ANGLE_LL = round(np.degrees(getAngle(a, b, c)), 2)
    
    if IP_LL_Z>IP_CP_Z:
        am1 = AngleAnnotation(center, p1[1], p2[1], ax=ax0, size=100, text=ANGLE_LL, linewidth=3, zorder=10)
    else:
        am1 = AngleAnnotation(center, p2[1], p1[1], ax=ax0, size=100, text=ANGLE_LL, linewidth=3, zorder=10)

    # OPENING DEVIATION
    center = (IP_CP_X, IP_CP_Z)
    p1 = np.array([(IP_CP_X, IP_CP_Z), (IP_MO_X, IP_MO_Z)])
    p2 = np.array([(IP_CP_X, IP_CP_Z), (IP_CP_X, IP_MO_Z)])

    a=np.array([IP_MO_X, IP_MO_Z])
    b=np.array([IP_CP_X, IP_CP_Z])
    c=np.array([IP_CP_X, IP_MO_Z])
    ANGLE_OD = round(np.degrees(getAngle(a, b, c)), 2)
    if IP_MO_X<IP_CP_X:
        am3 = AngleAnnotation(center, p1[1], p2[1], ax=ax0, size=275, text=ANGLE_OD, linewidth=3, zorder=10)
        OD_side=str('right')
    else:
        am3 = AngleAnnotation(center, p2[1], p1[1], ax=ax0, size=275, text=ANGLE_OD, linewidth=3, zorder=10)
        OD_side=str('left')
    print(OD_side)
    # SAGITTAL PLANE############################################################################
    ax1 = fig.add_subplot(spec[0:2, 2:4])
    ax1.patch.set_facecolor("whitesmoke")
    plt.gca().invert_xaxis()
    ax1.axis('equal')
    ax1.set_title("Sagittal Plane (incisal point)")
    ax1.set(xlabel="Y axis, mm", ylabel="Z axis, mm")
    ax1.xaxis.set_major_locator(MultipleLocator(5))
    ax1.yaxis.set_major_locator(MultipleLocator(5))
    ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax1.grid(which="minor", color="#e4e4e4", linestyle="--")
    ax1.grid(which="major", color="#CCCCCC", linestyle="--")
    ax1.plot(IP_Y_Array, IP_Z_Array, color="dimgray", linewidth=0.5)
    ax1.grid(True)
    ax1.axhline(y=IP_CP_Z, color="black", linestyle="--", linewidth=2)
    ax1.axvline(x=IP_CP_Y, color="black", linestyle="--", linewidth=2)
    ax1.plot(
        LineMaxOpenY,
        LineMaxOpenZ,
        color="b",
        linestyle="--",
        linewidth=3,
        marker="o",
        mfc="black",
        mec="black",
        ms=2,
    )
    ax1.plot(
        LineOpenY,
        LineOpenZ,
        color="red",
        linewidth=3,
        marker="o",
        mfc="black",
        mec="black",
        ms=6,
    )
    ax1.plot(
        LineProtrusionY,
        LineProtrusionZ,
        color="blue",
        linewidth=3,
        marker="o",
        mfc="black",
        mec="black",
        ms=6,
    )
    ax1.plot(
        LineLatRightY,
        LineLatRightZ,
        color="orange",
        linewidth=3,
        marker="o",
        mfc="black",
        mec="black",
        ms=6,
    )
    ax1.plot(
        LineLatLeftY,
        LineLatLeftZ,
        color="magenta",
        linewidth=3,
        marker="o",
        mfc="black",
        mec="black",
        ms=6,
    )

    ax = fig.add_subplot(spec[1, 3])
    image = plt.imread(join(ImgFolder, "incright.png"))
    ax.imshow(image, alpha=headimgalpha)
    ax.axis("off")

    # ANGLES

    # PROTRUSION
    center = (IP_CP_Y, IP_CP_Z)
    p1 = np.array([(IP_CP_Y, IP_CP_Z), (IP_PP_Y, IP_PP_Z)])
    p2 = np.array([(IP_CP_Y, IP_CP_Z), (IP_PP_Y, IP_CP_Z)])

    a=np.array([IP_PP_Y, IP_PP_Z])
    b=np.array([IP_CP_Y, IP_CP_Z])
    c=np.array([IP_PP_Y, IP_CP_Z])
    ANGLE_Rrot = round(np.degrees(getAngle(a, b, c)), 2)
    if IP_PP_Z<IP_CP_Z:
        am4 = AngleAnnotation(center, p1[1], p2[1], ax=ax1, size=100, text=ANGLE_Rrot, linewidth=3, zorder=10)
    else:
        am4 = AngleAnnotation(center, p2[1], p1[1], ax=ax1, size=100, text=ANGLE_Rrot, linewidth=3, zorder=10)



    # INFO PLOTS########################################################################

    MaxOpen = abs(round((IP_MO_Z) - (IP_CP_Z), 2))

    axI1 = fig.add_subplot(infospec[4, 0])
    axI1.grid(False)
    plt.axis("off")
    axI1.text(
        0.0,
        1.2,
        "Laterotrusion Right = "
        + str(ANGLE_LR)
        + "°"
        + "\n"
        + "Laterotrusion Left = "
        + str(ANGLE_LL)
        + "°"
        + "\n"
        + "Opening deviation = "
        + OD_side
        + " "
        + str(ANGLE_OD)
        + "°",
    )

    axI2 = fig.add_subplot(infospec[4, 2])
    axI2.grid(False)
    plt.axis("off")
    axI2.text(
        0.0,
        1.2,
        "Incisal point protrusion = "
        + str(ANGLE_Rrot)
        + "°"
        + "\n"
        + "Maximum opening = "
        + str(MaxOpen)
        + " mm"
        + "\n",
    )

    ###################################################################################
    # CONDYLES LINES
    ###################################################################################
    RC_CP_X, RC_CP_Y, RC_CP_Z = 0, 0, 0
    LC_CP_X, LC_CP_Y, LC_CP_Z = 0, 0, 0

    # LEFT CONDYLE
    LC_LineOpenX = [LC_CP_X, LC_MO_X]
    LC_LineOpenY = [LC_CP_Y, LC_MO_Y]
    LC_LineOpenZ = [LC_CP_Z, LC_MO_Z]

    LC_LineProtrusionX = [LC_CP_X, LC_PP_X]
    LC_LineProtrusionY = [LC_CP_Y, LC_PP_Y]
    LC_LineProtrusionZ = [LC_CP_Z, LC_PP_Z]

    LC_LineLatRightX = [LC_CP_X, LC_LR_X]
    LC_LineLatRightY = [LC_CP_Y, LC_LR_Y]
    LC_LineLatRightZ = [LC_CP_Z, LC_LR_Z]

    LC_LineLatLeftX = [LC_CP_X, LC_LL_X]
    LC_LineLatLeftY = [LC_CP_Y, LC_LL_Y]
    LC_LineLatLeftZ = [LC_CP_Z, LC_LL_Z]

    LC_LineMomShiftX = [LC_CP_X, LC_LL_X]
    LC_LineMomShiftY = [LC_CP_Y, LC_CP_Y]

    # RIGHT CONDYLE
    RC_LineOpenX = [RC_CP_X, RC_MO_X]
    RC_LineOpenY = [RC_CP_Y, RC_MO_Y]
    RC_LineOpenZ = [RC_CP_Z, RC_MO_Z]

    RC_LineProtrusionX = [RC_CP_X, RC_PP_X]
    RC_LineProtrusionY = [RC_CP_Y, RC_PP_Y]
    RC_LineProtrusionZ = [RC_CP_Z, RC_PP_Z]

    RC_LineLatRightX = [RC_CP_X, RC_LR_X]
    RC_LineLatRightY = [RC_CP_Y, RC_LR_Y]
    RC_LineLatRightZ = [RC_CP_Z, RC_LR_Z]

    RC_LineLatLeftX = [RC_CP_X, RC_LL_X]
    RC_LineLatLeftY = [RC_CP_Y, RC_LL_Y]
    RC_LineLatLeftZ = [RC_CP_Z, RC_LL_Z]

    RC_LineMomShiftX = [RC_CP_X, RC_LR_X]
    RC_LineMomShiftY = [RC_CP_Y, RC_CP_Y]

    ################################################################################################
    #    CONDYLE PLOTS
    ################################################################################################

    # RIGHT COND SAGITTAL PLANE
    ax2 = fig.add_subplot(spec[2:4, 0:2])
    ax2.axis('equal')
    plt.gca().invert_xaxis()
    ax2.xaxis.set_major_locator(MultipleLocator(5))
    ax2.yaxis.set_major_locator(MultipleLocator(5))
    ax2.xaxis.set_minor_locator(MultipleLocator(1))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))
    ax2.grid(which="minor", color="#b8b8b8", linestyle="--")
    ax2.grid(which="major", color="#959595", linestyle="--")

    ax2.patch.set_facecolor("lightskyblue")
    ax2.set_title("Sagittal Plane (right condile)")
    ax2.set(xlabel="Y axis, mm", ylabel="Z axis, mm")
    ax2.plot(RC_Y_Array, RC_Z_Array, color="dimgray", linewidth=0.7)
    ax2.grid(True)
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=2)
    ax2.plot(
        RC_LineOpenY,
        RC_LineOpenZ,
        color="red",
        linewidth=3,
        marker="o",
        mfc="black",
        mec="black",
        ms=6,
    )
    ax2.plot(
        RC_LineProtrusionY,
        RC_LineProtrusionZ,
        color="blue",
        linewidth=3,
        marker="o",
        mfc="black",
        mec="black",
        ms=6,
    )
    ax2.plot(
        RC_LineLatRightY,
        RC_LineLatRightZ,
        color="orange",
        linewidth=3,
        marker="o",
        mfc="black",
        mec="black",
        ms=6,
    )
    ax2.plot(
        RC_LineLatLeftY,
        RC_LineLatLeftZ,
        color="magenta",
        linewidth=3,
        marker="o",
        mfc="black",
        mec="black",
        ms=6,
    )


    #ANGLE OF CONDYLAR GUIDANCE
    center = (RC_CP_Y, RC_CP_Z)
    p1 = np.array([(RC_CP_Y, RC_CP_Z), (RC_PP_Y, RC_CP_Z)])
    p2 = np.array([(RC_CP_Y, RC_CP_Z), (RC_PP_Y, RC_PP_Z)])

    a=np.array([RC_PP_Y, RC_CP_Z])
    b=np.array([RC_CP_Y, RC_CP_Z])
    c=np.array([RC_PP_Y, RC_PP_Z])
    ANGLE_RC_Rrot = round(np.degrees(getAngle(a, b, c)), 2)
    if RC_PP_Z>RC_CP_Z:
        am5 = AngleAnnotation(center, p1[1], p2[1], ax=ax2, size=200, text=ANGLE_RC_Rrot, linewidth=3, zorder=10)
    else:
        am5 = AngleAnnotation(center, p2[1], p1[1], ax=ax2, size=200, text=ANGLE_RC_Rrot, linewidth=3, zorder=10)


    ax = fig.add_subplot(spec[3, 0])
    image = plt.imread(join(ImgFolder, "rightright.png"))
    ax.imshow(image, alpha=headimgalpha)
    ax.axis("off")

    ################################################################################################
    # LEFT COND SAGITTAL PLANE
    ax3 = fig.add_subplot(spec[2:4, 2:4])
    ax3.axis('equal')
    ax3.xaxis.set_major_locator(MultipleLocator(5))
    ax3.yaxis.set_major_locator(MultipleLocator(5))
    ax3.xaxis.set_minor_locator(MultipleLocator(1))
    ax3.yaxis.set_minor_locator(MultipleLocator(1))
    ax3.grid(which="minor", color="#b8b8b8", linestyle="--")
    ax3.grid(which="major", color="#959595", linestyle="--")

    ax3.patch.set_facecolor("antiquewhite")
    ax3.set_title("Sagittal Plane (left condile)")
    ax3.set(xlabel="Y axis, mm", ylabel="Z axis, mm")
    ax3.plot(LC_Y_Array, LC_Z_Array, color="dimgray", linewidth=0.7)
    ax3.grid(True)
    ax3.axhline(y=0, color="black", linestyle="--", linewidth=2)
    ax3.plot(
        LC_LineOpenY,
        LC_LineOpenZ,
        color="red",
        linewidth=3,
        marker="o",
        mfc="black",
        mec="black",
        ms=6,
    )
    ax3.plot(
        LC_LineProtrusionY,
        LC_LineProtrusionZ,
        color="blue",
        linewidth=3,
        marker="o",
        mfc="black",
        mec="black",
        ms=6,
    )
    ax3.plot(
        LC_LineLatRightY,
        LC_LineLatRightZ,
        color="orange",
        linewidth=3,
        marker="o",
        mfc="black",
        mec="black",
        ms=6,
    )
    ax3.plot(
        LC_LineLatLeftY,
        LC_LineLatLeftZ,
        color="magenta",
        linewidth=3,
        marker="o",
        mfc="black",
        mec="black",
        ms=6,
    )

    # ANGLE OF CONDYLAR GUIDANCE
    center=(LC_CP_Y, LC_CP_Z)
    p1 = np.array([(LC_CP_Y, LC_CP_Z), (LC_PP_Y, LC_CP_Z)])
    p2 = np.array([(LC_CP_Y, LC_CP_Z), (LC_PP_Y, LC_PP_Z)])
    a=np.array([LC_PP_Y, LC_CP_Z])
    b=np.array([LC_CP_Y, LC_CP_Z])
    c=np.array([LC_PP_Y, LC_PP_Z])
    ANGLE_LC_Lrot = round(np.degrees(getAngle(a, b, c)), 2)
    if LC_PP_Z<LC_CP_Z:
        am6 = AngleAnnotation(center, p1[1], p2[1], ax=ax3, size=200, text=ANGLE_LC_Lrot, linewidth=3, zorder=10)
    else:
        am6 = AngleAnnotation(center, p2[1], p1[1], ax=ax3, size=200, text=ANGLE_LC_Lrot, linewidth=3, zorder=10)


    ax = fig.add_subplot(spec[3, 3])
    image = plt.imread(join(ImgFolder, "leftleft.png"))
    ax.imshow(image, alpha=headimgalpha)
    ax.axis("off")


    # INFO PLOTS########################################################################
    axI3 = fig.add_subplot(infospec[7, 0])
    axI3.grid(False)
    plt.axis("off")

    axI3.text(0.0, -0.6, "RIGHT TMJ" + "\n" + "Angle of condylar guidance = " + str(ANGLE_RC_Rrot) + "°")

    axI3 = fig.add_subplot(infospec[7, 2])
    axI3.grid(False)
    plt.axis("off")
    axI3.text(0.0, -0.6, "LEFT TMJ" + "\n" + "Angle of condylar guidance = " + str(ANGLE_LC_Lrot) + "°")


    ################################################################################################
    # RIGHT COND TRANSVERCE PLANE
    ax4 = fig.add_subplot(spec[4:6, 0:2])
    ax4.axis('equal')
    ax4.xaxis.set_major_locator(MultipleLocator(5))
    ax4.yaxis.set_major_locator(MultipleLocator(5))
    ax4.xaxis.set_minor_locator(MultipleLocator(1))
    ax4.yaxis.set_minor_locator(MultipleLocator(1))
    ax4.grid(which="minor", color="#b8b8b8", linestyle="--")
    ax4.grid(which="major", color="#959595", linestyle="--")

    ax4.patch.set_facecolor("lightskyblue")
    ax4.set_title("Axial Plane (right condile)")
    ax4.set(xlabel="X axis, mm", ylabel="Y axis, mm")
    ax4.plot(RC_X_Array, RC_Y_Array, color="dimgray", linewidth=0.7)
    ax4.grid(True)
    ax4.plot(
        RC_LineOpenX,
        RC_LineOpenY,
        color="red",
        linewidth=3,
        marker="o",
        mfc="black",
        mec="black",
        ms=6,
    )
    ax4.plot(
        RC_LineProtrusionX,
        RC_LineProtrusionY,
        color="blue",
        linewidth=3,
        marker="o",
        mfc="black",
        mec="black",
        ms=6,
    )
    ax4.plot(
        RC_LineLatRightX,
        RC_LineLatRightY,
        color="orange",
        linewidth=3,
        marker="o",
        mfc="black",
        mec="black",
        ms=6,
    )
    ax4.plot(
        RC_LineLatLeftX,
        RC_LineLatLeftY,
        color="magenta",
        linewidth=3,
        marker="o",
        mfc="black",
        mec="black",
        ms=6,
    )
    ax4.axhline(y=0, color="black", linestyle="--", linewidth=2)
    ax4.axvline(x=0, color="black", linestyle="--", linewidth=2)
    ax4.plot(
        RC_LineProtrusionX,
        RC_LineProtrusionY,
        color="blue",
        linewidth=3,
        marker="o",
        mfc="black",
        mec="black",
        ms=6,
    )
    ax4.plot(
        RC_LineMomShiftX,
        RC_LineMomShiftY,
        color="navy",
        linewidth=3,
        marker="D",
        mfc="white",
        mec="black",
        ms=6,
        markeredgewidth=2,
    )

    ax = fig.add_subplot(spec[5, 0])
    image = plt.imread(join(ImgFolder, "righttop.png"))
    ax.imshow(image, alpha=headimgalpha)
    ax.axis("off")

    # BENNETT ANGLE
    center = (RC_CP_X, RC_CP_Y)
    p1 = np.array([(RC_CP_X, RC_CP_Y), (RC_LL_X, RC_LL_Y)])
    p2 = np.array([(RC_CP_X, RC_CP_Y), (RC_CP_X, RC_PP_Y)])

    a=np.array([RC_LL_X, RC_LL_Y])
    b=np.array([RC_CP_X, RC_CP_Y])
    c=np.array([RC_CP_X, RC_PP_Y])
    ANGLE_RBen = round(np.degrees(getAngle(a, b, c)), 2)
    if RC_LL_X<RC_CP_X:
        am7 = AngleAnnotation(center, p1[1], p2[1], ax=ax4, size=200, text=ANGLE_RBen, linewidth=3, zorder=10)
    else:
        am7 = AngleAnnotation(center, p2[1], p1[1], ax=ax4, size=200, text=ANGLE_RBen, linewidth=3, zorder=10)



    ################################################################################################
    # LEFT COND TRANSVERCE PLANE
    ax5 = fig.add_subplot(spec[4:6, 2:4])
    ax5.axis('equal')
    ax5.xaxis.set_major_locator(MultipleLocator(5))
    ax5.yaxis.set_major_locator(MultipleLocator(5))
    ax5.xaxis.set_minor_locator(MultipleLocator(1))
    ax5.yaxis.set_minor_locator(MultipleLocator(1))
    ax5.grid(which="minor", color="#b8b8b8", linestyle="--")
    ax5.grid(which="major", color="#959595", linestyle="--")

    ax5.patch.set_facecolor("antiquewhite")
    ax5.set_title("Axial Plane (left condile)")
    ax5.set(xlabel="X axis, mm", ylabel="Y axis, mm")
    ax5.plot(LC_X_Array, LC_Y_Array, color="dimgray", linewidth=0.7)
    ax5.grid(True)
    ax5.plot(
        LC_LineOpenX,
        LC_LineOpenY,
        color="red",
        linewidth=3,
        marker="o",
        mfc="black",
        mec="black",
        ms=6,
    )
    ax5.plot(
        LC_LineProtrusionX,
        LC_LineProtrusionY,
        color="blue",
        linewidth=3,
        marker="o",
        mfc="black",
        mec="black",
        ms=6,
    )
    ax5.plot(
        LC_LineLatRightX,
        LC_LineLatRightY,
        color="orange",
        linewidth=3,
        marker="o",
        mfc="black",
        mec="black",
        ms=6,
    )
    ax5.plot(
        LC_LineLatLeftX,
        LC_LineLatLeftY,
        color="magenta",
        linewidth=3,
        marker="o",
        mfc="black",
        mec="black",
        ms=6,
    )
    ax5.axhline(y=0, color="black", linestyle="--", linewidth=2)
    ax5.axvline(x=0, color="black", linestyle="--", linewidth=2)

    ax5.plot(
        LC_LineMomShiftX,
        LC_LineMomShiftY,
        color="navy",
        linewidth=3,
        marker="D",
        mfc="white",
        mec="black",
        ms=6,
        markeredgewidth=2,
    )

    # BENNETT ANGLE
    center = (LC_CP_X, LC_CP_Y)
    p1 = np.array([(LC_CP_X, LC_CP_Y), (LC_LR_X, LC_LR_Y)])
    p2 = np.array([(LC_CP_X, LC_CP_Y), (LC_CP_X, LC_PP_Y)])

    a=np.array([LC_LR_X, LC_LR_Y])
    b=np.array([LC_CP_X, LC_CP_Y])
    c=np.array([LC_CP_X, LC_PP_Y])
    ANGLE_LBen = round(np.degrees(getAngle(a, b, c)), 2)
    if LC_LR_X<LC_CP_X:
        am8 = AngleAnnotation(center, p1[1], p2[1], ax=ax5, size=200, text=ANGLE_LBen, linewidth=3, zorder=10)
    else:
        am8 = AngleAnnotation(center, p2[1], p1[1], ax=ax5, size=200, text=ANGLE_LBen, linewidth=3, zorder=10)


    ax = fig.add_subplot(spec[5, 3])
    image = plt.imread(join(ImgFolder, "lefttop.png"))
    ax.imshow(image, alpha=headimgalpha)
    ax.axis("off")

    # INFO PLOTS########################################################################
    RightShift = abs(round(RC_LR_X - RC_CP_X, 3))
    LeftShift = abs(round(LC_LL_X - LC_CP_X, 3))

    axI5 = fig.add_subplot(infospec[11, 0])
    axI5.grid(False)
    plt.axis("off")
    axI5.text(
        0.0,
        -1,
        "RIGHT TMJ"
        + "\n"
        + "Bennett angle = "
        + str(ANGLE_RBen)
        + "°"        
        + "\n"
        + "Bennett shift = "
        + str(RightShift)
        + " mm",
    )

    axI6 = fig.add_subplot(infospec[11, 2])
    axI6.grid(False)
    plt.axis("off")
    axI6.text(
        0.0,
        -1,
        "LEFT TMJ"
        + "\n"
        + "Bennett angle = "
        + str(ANGLE_LBen)
        + "°"
        + "\n"
        + "Bennett shift = "
        + str(LeftShift)
        + " mm",
    )

    return fig

def getAngle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle


class AngleAnnotation(Arc):
    """
    Draws an arc between two vectors which appears circular in display space.
    """
    def __init__(self, xy, p1, p2, size=75, unit="points", ax=None,
                 text="", textposition="outside", text_kw=None, **kwargs):
        
        self.ax = ax or plt.gca()
        self._xydata = xy  # in data coordinates
        self.vec1 = p1
        self.vec2 = p2
        self.size = size
        self.unit = unit
        self.textposition = textposition

        super().__init__(self._xydata, size, size, angle=0.0,
                         theta1=self.theta1, theta2=self.theta2, **kwargs)

        self.set_transform(IdentityTransform())
        self.ax.add_patch(self)

        self.kw = dict(ha="center", va="center",
                       xycoords=IdentityTransform(),
                       xytext=(0, 0), textcoords="offset points",
                       annotation_clip=True)
        self.kw.update(text_kw or {})
        self.text = ax.annotate(text, xy=self._center, **self.kw)

    def get_size(self):
        factor = 1.
        if self.unit == "points":
            factor = self.ax.figure.dpi / 72.
        elif self.unit[:4] == "axes":
            b = TransformedBbox(Bbox.from_bounds(0, 0, 1, 1),
                                self.ax.transAxes)
            dic = {"max": max(b.width, b.height),
                   "min": min(b.width, b.height),
                   "width": b.width, "height": b.height}
            factor = dic[self.unit[5:]]
        return self.size * factor

    def set_size(self, size):
        self.size = size

    def get_center_in_pixels(self):
        """return center in pixels"""
        return self.ax.transData.transform(self._xydata)

    def set_center(self, xy):
        """set center in data coordinates"""
        self._xydata = xy

    def get_theta(self, vec):
        vec_in_pixels = self.ax.transData.transform(vec) - self._center
        return np.rad2deg(np.arctan2(vec_in_pixels[1], vec_in_pixels[0]))

    def get_theta1(self):
        return self.get_theta(self.vec1)

    def get_theta2(self):
        return self.get_theta(self.vec2)

    def set_theta(self, angle):
        pass

    # Redefine attributes of the Arc to always give values in pixel space
    _center = property(get_center_in_pixels, set_center)
    theta1 = property(get_theta1, set_theta)
    theta2 = property(get_theta2, set_theta)
    width = property(get_size, set_size)
    height = property(get_size, set_size)

    # The following two methods are needed to update the text position.
    def draw(self, renderer):
        self.update_text()
        super().draw(renderer)

    def update_text(self):
        c = self._center
        s = self.get_size()
        angle_span = (self.theta2 - self.theta1) % 360
        angle = np.deg2rad(self.theta1 + angle_span / 2)
        r = s / 2
        if self.textposition == "inside":
            r = s / np.interp(angle_span, [60, 90, 135, 180],
                                          [3.3, 3.5, 3.8, 4])
        self.text.xy = c + r * np.array([np.cos(angle), np.sin(angle)])
        if self.textposition == "outside":
            def R90(a, r, w, h):
                if a < np.arctan(h/2/(r+w/2)):
                    return np.sqrt((r+w/2)**2 + (np.tan(a)*(r+w/2))**2)
                else:
                    c = np.sqrt((w/2)**2+(h/2)**2)
                    T = np.arcsin(c * np.cos(np.pi/2 - a + np.arcsin(h/2/c))/r)
                    xy = r * np.array([np.cos(a + T), np.sin(a + T)])
                    xy += np.array([w/2, h/2])
                    return np.sqrt(np.sum(xy**2))

            def R(a, r, w, h):
                aa = (a % (np.pi/4))*((a % (np.pi/2)) <= np.pi/4) + \
                     (np.pi/4 - (a % (np.pi/4)))*((a % (np.pi/2)) >= np.pi/4)
                return R90(aa, r, *[w, h][::int(np.sign(np.cos(2*a)))])

            bbox = self.text.get_window_extent()
            X = R(angle, r, bbox.width, bbox.height)
            trans = self.ax.figure.dpi_scale_trans.inverted()
            offs = trans.transform(((X-s/2), 0))[0] * 72
            self.text.set_position([offs*np.cos(angle), offs*np.sin(angle)])


class MyApp(QWidget):
    def __init__(self, fig):
        super().__init__()
        self.title = "Paths Report"
        self.posXY = (10, 40)
        self.windowSize = (1188, 1000)
        self.fig = fig

    def initUI(self):
        QMainWindow().setCentralWidget(QWidget())

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        canvas = FigCanvas(self.fig)
        canvas.draw()

        scroll = QScrollArea(self)
        scroll.setWidget(canvas)

        nav = NabToolbar(canvas, self)
        self.layout().addWidget(nav)
        self.layout().addWidget(scroll)

        self.show_basic()

    def show_basic(self):
        self.setWindowTitle(self.title)
        self.setGeometry(*self.posXY, *self.windowSize)
        self.show()


def halfangle(a, b):
    "Gets the middle angle between a and b, when increasing from a to b"
    if b < a:
        b += 360
    return (a + b) / 2 % 360


def GetEmptyMovementsArray(EmptyName, FrameStart=None, FrameEnd=None):

    Obj = bpy.data.objects.get(EmptyName)
    if Obj:
        if not FrameStart:
            FrameStart = Obj.motion_path.frame_start
        if not FrameEnd:
            FrameEnd = Obj.motion_path.frame_end

        Points = Obj.motion_path.points
        PointsCoordsList = []
        for F in range(FrameStart, FrameEnd):
            PointsCoordsList.append(list(Points[F].co))

        return np.array(PointsCoordsList)
    else:
        print(f"No object named {EmptyName}")
        return None


def GetPeakPointCouples(IP, RCP, LCP, DIST):

    if bpy.context.scene.render.fps >= 40:
        DIST = 100
    else:
        DIST = 50

    print(DIST)
    LC_Array = GetEmptyMovementsArray(EmptyName=LCP.name)
    LC_X_Array, LC_Y_Array, LC_Z_Array = (
        LC_Array[:, 0],
        LC_Array[:, 1],
        LC_Array[:, 2],
    )

    RC_Array = GetEmptyMovementsArray(EmptyName=RCP.name)
    RC_X_Array, RC_Y_Array, RC_Z_Array = (
        RC_Array[:, 0],
        RC_Array[:, 1],
        RC_Array[:, 2],
    )

    IP_Array = GetEmptyMovementsArray(EmptyName=IP.name)
    IP_X_Array, IP_Y_Array, IP_Z_Array = IP_Array[:, 0], IP_Array[:, 1], IP_Array[:, 2]

    IP_X_Min, IP_Y_Min, IP_Z_Min = np.min(IP_Array, axis=0)
    IP_X_Max, IP_Y_Max, IP_Z_Max = np.max(IP_Array, axis=0)
   #IP_X_Averrage, IP_Y_Averrage, IP_Z_Averrage = np.mean(IP_Array, axis=0)

    ###################################################################################
    # PEAKS
    ###################################################################################
    peaksX_Right, _ = find_peaks(
        -IP_X_Array, height=(-IP_X_Min + IP_X_Min / 5, -IP_X_Min), distance=DIST
    )
    peaksX_Left, _ = find_peaks(
        IP_X_Array, height=(IP_X_Max - IP_X_Max / 5, IP_X_Max), distance=DIST
    )
    peaksY_Protrusion, _ = find_peaks(
        -IP_Y_Array, height=(-IP_Y_Min + IP_Y_Min / 40, -IP_Y_Min), distance=DIST
    )
    peaksY_Open, _ = find_peaks(
        IP_Y_Array, height=(IP_Y_Max + IP_Y_Max / 40, IP_Y_Max), distance=DIST
    )
    #peaksZ_Open, _ = find_peaks(
    #    -IP_Z_Array, height=(-IP_Z_Min + IP_Z_Min / 40, -IP_Z_Min), distance=DIST
    #)
    peaksZ_Central, _ = find_peaks(
        IP_Z_Array, height=(IP_Z_Max + IP_Z_Max / 40, IP_Z_Max), distance=DIST
    )
    ###################################################################################
    # POINTS
    ###################################################################################

    PointsList = []

    # CENTRAL : ##################################################

    IP_Central = [
        np.mean(IP_X_Array[peaksZ_Central]),
        np.mean(IP_Y_Array[peaksZ_Central]),
        np.mean(IP_Z_Array[peaksZ_Central]),
    ]

    RC_Central = [
        np.mean(RC_X_Array[peaksZ_Central]),
        np.mean(RC_Y_Array[peaksZ_Central]),
        np.mean(RC_Z_Array[peaksZ_Central]),
    ]

    LC_Central = [
        np.mean(LC_X_Array[peaksZ_Central]),
        np.mean(LC_Y_Array[peaksZ_Central]),
        np.mean(LC_Z_Array[peaksZ_Central]),
    ]

    # PROTRUSION : ###################################

    IP_Protrusion = [
        np.mean(IP_X_Array[peaksY_Protrusion]),
        np.mean(IP_Y_Array[peaksY_Protrusion]),
        np.mean(IP_Z_Array[peaksY_Protrusion]),
    ]

    RC_Protrusion = [
        np.mean(RC_X_Array[peaksY_Protrusion]),
        np.mean(RC_Y_Array[peaksY_Protrusion]),
        np.mean(RC_Z_Array[peaksY_Protrusion]),
    ]

    LC_Protrusion = [
        np.mean(LC_X_Array[peaksY_Protrusion]),
        np.mean(LC_Y_Array[peaksY_Protrusion]),
        np.mean(LC_Z_Array[peaksY_Protrusion]),
    ]

    PointsList.append([("IP-Protrusion", IP_Protrusion), ("IP_Central", IP_Central)])
    PointsList.append([("RC-Protrusion", RC_Protrusion), ("RC_Central", RC_Central)])
    PointsList.append([("LC-Protrusion", LC_Protrusion), ("LC_Central", LC_Central)])

    # LATERAL RIGHT : ###################################

    IP_Lateral_Right = [
        np.mean(IP_X_Array[peaksX_Right]),
        np.mean(IP_Y_Array[peaksX_Right]),
        np.mean(IP_Z_Array[peaksX_Right]),
    ]

    RC_Lateral_Right = [
        np.mean(RC_X_Array[peaksX_Right]),
        np.mean(RC_Y_Array[peaksX_Right]),
        np.mean(RC_Z_Array[peaksX_Right]),
    ]

    LC_Lateral_Right = [
        np.mean(LC_X_Array[peaksX_Right]),
        np.mean(LC_Y_Array[peaksX_Right]),
        np.mean(LC_Z_Array[peaksX_Right]),
    ]

    PointsList.append(
        [("IP_Lateral-Right", IP_Lateral_Right), ("IP_Central", IP_Central)]
    )
    PointsList.append(
        [("RC_Lateral-Right", RC_Lateral_Right), ("RC_Central", RC_Central)]
    )
    PointsList.append(
        [("LC_Lateral-Right", LC_Lateral_Right), ("LC_Central", LC_Central)]
    )

    # LATERAL LEFT : ###################################

    IP_Lateral_Left = [
        np.mean(IP_X_Array[peaksX_Left]),
        np.mean(IP_Y_Array[peaksX_Left]),
        np.mean(IP_Z_Array[peaksX_Left]),
    ]

    RC_Lateral_Left = [
        np.mean(RC_X_Array[peaksX_Left]),
        np.mean(RC_Y_Array[peaksX_Left]),
        np.mean(RC_Z_Array[peaksX_Left]),
    ]

    LC_Lateral_Left = [
        np.mean(LC_X_Array[peaksX_Left]),
        np.mean(LC_Y_Array[peaksX_Left]),
        np.mean(LC_Z_Array[peaksX_Left]),
    ]

    PointsList.append(
        [("IP_Lateral-Left", IP_Lateral_Left), ("IP_Central", IP_Central)]
    )
    PointsList.append(
        [("RC_Lateral-Left", RC_Lateral_Left), ("RC_Central", RC_Central)]
    )
    PointsList.append(
        [("LC_Lateral-Left", LC_Lateral_Left), ("LC_Central", LC_Central)]
    )

    # MAXIMUM OPENING : ###################################

    IP_Max_Open = [
        np.mean(IP_X_Array[peaksY_Open]),
        np.mean(IP_Y_Array[peaksY_Open]),
        np.mean(IP_Z_Array[peaksY_Open]),
    ]

    RC_Max_Open = [
        np.mean(RC_X_Array[peaksY_Open]),
        np.mean(RC_Y_Array[peaksY_Open]),
        np.mean(RC_Z_Array[peaksY_Open]),
    ]

    LC_Max_Open = [
        np.mean(LC_X_Array[peaksY_Open]),
        np.mean(LC_Y_Array[peaksY_Open]),
        np.mean(LC_Z_Array[peaksY_Open]),
    ]

    PointsList.append([("IP_Max-Open", IP_Max_Open), ("IP_Central", IP_Central)])
    PointsList.append([("RC_Max-Open", RC_Max_Open), ("RC_Central", RC_Central)])
    PointsList.append([("LC_Max-Open", LC_Max_Open), ("LC_Central", LC_Central)])

    return PointsList


def AddHookedSegment(Points, Name, color, thikness, CollName=None):
    bpy.ops.curve.primitive_bezier_curve_add(
        radius=1, enter_editmode=False, align="CURSOR"
    )
    bpy.ops.object.mode_set(mode="OBJECT")
    Segment = bpy.context.view_layer.objects.active
    Segment.name = Name
    Segment.data.name = Name

    # Add color material :
    SegmentMat = bpy.data.materials.get(f"{Name}_Mat") or bpy.data.materials.new(
        f"{Name}_Mat"
    )
    SegmentMat.diffuse_color = color
    Segment.active_material = SegmentMat

    SegmentPoints = Segment.data.splines[0].bezier_points[:]
    SegmentPoints[0].co = Segment.matrix_world.inverted() @ Points[0].location
    SegmentPoints[1].co = Segment.matrix_world.inverted() @ Points[1].location

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.curve.select_all(action="SELECT")
    bpy.ops.curve.handle_type_set(type="VECTOR")
    bpy.context.object.data.bevel_depth = thikness / 2

    # Hook Segment to spheres
    for i, P in enumerate(Points):
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        P.select_set(True)
        Segment.select_set(True)
        bpy.context.view_layer.objects.active = Segment
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.curve.select_all(action="DESELECT")
        SegmentPoints = Segment.data.splines[0].bezier_points[:]
        SegmentPoints[i].select_control_point = True
        bpy.ops.object.hook_add_selob(use_bone=False)

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    Segment.hide_select = True
    MoveToCollection(Segment, CollName)


# Add Emptys
def AddEmpty(type, name, location, radius, CollName=None):
    bpy.ops.object.empty_add(type=type, radius=radius, location=location)
    Empty = bpy.context.object
    Empty.name = name
    if CollName:
        MoveToCollection(Empty, CollName)
    return Empty


###############################################################
# GPU, Blf
##########################################################
def AddGpuPoints(PcoList, colors, Thikness):
    def draw(Thikness):
        bgl.glLineWidth(Thikness)
        shader.bind()
        batch.draw(shader)
        bgl.glLineWidth(1)

    shader = gpu.shader.from_builtin("3D_SMOOTH_COLOR")
    batch = batch_for_shader(shader, "POINTS", {"pos": PcoList, "color": colors})
    _Handler = bpy.types.SpaceView3D.draw_handler_add(
        draw, (Thikness,), "WINDOW", "POST_VIEW"
    )

    for area in bpy.context.window.screen.areas:
        if area.type == "VIEW_3D":
            area.tag_redraw()

    return _Handler


def Add_2D_BlfText(
    Font_Path, color=[1.0, 0.1, 0.0, 1.0], horiz=20, vert=40, size=50, text="BDENTAL-4D"
):

    font_id = 0

    def draw_callback_px(self, context):

        blf.color(font_id, color[0], color[1], color[2], color[3])
        blf.position(font_id, horiz, vert, 0)
        blf.size(font_id, size, 72)
        blf.draw(font_id, text)

    if Font_Path:
        if os.path.exists(Font_Path):
            font_id = blf.load(Font_Path)

    _Handler = bpy.types.SpaceView3D.draw_handler_add(
        draw_callback_px, (None, None), "WINDOW", "POST_PIXEL"
    )  # 2D :'POST_PIXEL' | 3D :'POST_VIEW'

    for area in bpy.context.window.screen.areas:
        if area.type == "VIEW_3D":
            area.tag_redraw()

    return _Handler


###################################################################
def Angle(v1, v2):
    dot_product = v1.normalized().dot(v2.normalized())
    Angle = degrees(acos(dot_product))
    return Angle


def Linked_Edges_Verts(v, mesh):
    Edges = [e for e in mesh.edges if v.index in e.vertices]
    Link_Verts = [
        mesh.vertices[idx] for e in Edges for idx in e.vertices if idx != v.index
    ]
    return Edges, Link_Verts


def ShortPath2(obj, Vid_List, close=True):
    mesh = obj.data
    zipList = list(zip(Vid_List, Vid_List[1:] + [Vid_List[0]]))

    Tuples = zipList
    if not close:
        Tuples = zipList[:-1]
    LoopIds = []
    for i, t in enumerate(Tuples):
        v0, v1 = mesh.vertices[t[0]], mesh.vertices[t[1]]
        LoopIds.append(v0.index)

        while True:
            CurrentID = LoopIds[-1]
            
            V_current = mesh.vertices[CurrentID]
            TargetVector = v1.co - V_current.co
            edges, verts = Linked_Edges_Verts(V_current, mesh)
            if verts:
                if v1 in verts:
                    LoopIds.append(v1.index)
                    break
                else:

                    v = min(
                        [
                            (abs(Angle(v.co - V_current.co, TargetVector)), v)
                            for v in verts
                        ]
                    )[1]
                    LoopIds.append(v.index)
                    print(v.index)
            else:
                break

    return LoopIds

def ShortestPath(obj, VidList, close=True) :
    
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    bpy.ops.mesh.select_all(action='DESELECT') 
    
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    bm.verts.ensure_lookup_table()
    
    Ids = VidList
    zipList = list(zip( Ids, Ids[1:]+[Ids[0]]))
    Ids_Tuples = zipList
    if not close:
        Ids_Tuples = zipList[:-1]
    
    Path = []
    
    for i,Ids in enumerate(Ids_Tuples):
        Path.append(Ids[0])
        bpy.ops.mesh.select_all(action='DESELECT')
        for id in Ids :
            bm.verts[id].select_set(True)
        select = [v.index for v in bm.verts if v.select]
        if len(select)>1:
            bpy.ops.mesh.shortest_path_select()
        select = [v.index for v in bm.verts if v.select]
        Path.extend(select)
        print(f'loop ({i}/{len(Ids_Tuples)}) processed ...')
       
            
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode="OBJECT")
    for id in Path :
        me.vertices[id].select = True
    CutLine = [v.index for v in me.vertices if v.select]
    print(f"selected verts : {len(CutLine)}")

    return CutLine
    
