import os, sys, time, bpy
import mathutils
import cv2
from cv2 import aruco

import gpu
from gpu_extras.batch import batch_for_shader
import bgl
import blf

from .BDENTAL_4D_Utils import *

#######################################################################################
########################### BDJawTracker WAXUP Operators ##############################
#######################################################################################


class BDENTAL_4D_OT_LowJawChild(bpy.types.Operator):
    """  Make LowJaw Child to LowMarker """

    bl_idname = "bdental4d.lowjawchildtolowmarker"
    bl_label = "Start Lower Jaw movings!"

    def execute(self, context):

        bpy.ops.object.select_all(action="DESELECT")
        LowMarker = bpy.data.objects["LowMarker"]
        LowJaw = bpy.data.objects["LowJaw"]
        LowJaw.select_set(True)
        bpy.context.view_layer.objects.active = LowJaw
        My_constraint = LowJaw.constraints.get("LowMarker_Child")

        if My_constraint:
            LowJaw.constraints.remove(My_constraint)
            bpy.ops.object.constraint_add(type="CHILD_OF")
            bpy.context.object.constraints["Child Of"].target = LowMarker
            bpy.context.object.constraints["Child Of"].name = "LowMarker_Child"

        else:
            bpy.ops.object.constraint_add(type="CHILD_OF")
            bpy.context.object.constraints["Child Of"].target = LowMarker
            bpy.context.object.constraints["Child Of"].name = "LowMarker_Child"

        return {"FINISHED"}


#######################################################################################
# Set Lower jaw Plane  Operator :
#######################################################################################


class BDENTAL_4D_OT_BakePlane(bpy.types.Operator):
    """ Will create and bake occlusal plane or planes"""

    bl_idname = "bdental4d.bakeplane"
    bl_label = "Bake occlusal plane/s"

    def execute(self, context):
        BDENTAL_4D_Props = context.scene.BDENTAL_4D_Props
        if BDENTAL_4D_Props.BakeUpPlane == True:
            self.bakeUpPlane()

        if BDENTAL_4D_Props.BakeLowPlane == True:
            self.bakeLowPlane()

        return {"FINISHED"}

    def bakeUpPlane(self):
        print("up")
        scn = bpy.context.scene
        frame_start = scn.frame_start
        frame_end = scn.frame_end

        LowJaw = bpy.data.objects["LowJaw"]
        UpJaw = bpy.data.objects["UpJaw"]

        for o in ("LowJaw", "UpJaw", "Occlusal_Plane"):
            obj = bpy.context.scene.objects.get(o)
            # if obj: obj.hide_viewport = False
            if obj:
                obj.hide_set(False)

        ###############################################################

        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.empty_add(
            type="PLAIN_AXES", align="WORLD", location=(0, 0, 0), scale=(10, 10, 10)
        )
        UpJawDriver = bpy.context.object
        UpJawDriver.name = "UpJawDriver"

        bpy.context.object.empty_display_size = 10
        bpy.ops.object.constraint_add(type="COPY_LOCATION")
        bpy.context.object.constraints["Copy Location"].target = LowJaw
        bpy.context.object.constraints["Copy Location"].invert_x = True
        bpy.context.object.constraints["Copy Location"].invert_y = True
        bpy.context.object.constraints["Copy Location"].invert_z = True
        bpy.ops.object.constraint_add(type="COPY_ROTATION")
        bpy.context.object.constraints["Copy Rotation"].target = LowJaw
        bpy.context.object.constraints["Copy Rotation"].invert_x = True
        bpy.context.object.constraints["Copy Rotation"].invert_y = True
        bpy.context.object.constraints["Copy Rotation"].invert_z = True

        bpy.ops.object.select_all(action="DESELECT")
        UpJaw.select_set(True)
        bpy.context.view_layer.objects.active = UpJaw

        bpy.ops.object.duplicate()
        UpJawMoved = bpy.context.object
        UpJawMoved.select_set(True)
        UpJawMoved.name = "UpJawMoved"
        UpJawDriver = bpy.data.objects["UpJawDriver"]
        bpy.ops.object.constraint_add(type="CHILD_OF")
        bpy.context.object.constraints["Child Of"].target = UpJawDriver

        #########################################################

        bpy.ops.object.select_all(action="DESELECT")
        Occlusal_Plane = bpy.data.objects["Occlusal_Plane"]
        Occlusal_Plane.select_set(True)

        bpy.context.view_layer.objects.active = Occlusal_Plane

        bpy.ops.object.duplicate()
        LowOcclPlane = bpy.context.object
        LowOcclPlane.select_set(True)
        LowOcclPlane.name = "UpOcclPlane"

        loc = LowOcclPlane.location
        (x, y, z) = (0.0, 0.0, 5.0)
        LowOcclPlane.location = loc + mathutils.Vector((x, y, z))

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.subdivide(number_cuts=6)
        bpy.ops.object.mode_set(mode="OBJECT")

        #########################################################
        # Shrinkwrap
        #########################################################

        UpJawMoved = bpy.data.objects["UpJawMoved"]
        scn = bpy.context.scene
        frame_start = scn.frame_start
        frame_end = scn.frame_end

        frame = frame_start

        while frame <= frame_end:
            bpy.context.scene.frame_set(frame)
            bpy.ops.object.modifier_add(type="SHRINKWRAP")
            bpy.context.object.modifiers["Shrinkwrap"].target = UpJawMoved
            bpy.context.object.modifiers["Shrinkwrap"].wrap_method = "PROJECT"
            bpy.context.object.modifiers["Shrinkwrap"].wrap_mode = "OUTSIDE"
            bpy.context.object.modifiers["Shrinkwrap"].use_project_z = True
            bpy.context.object.modifiers["Shrinkwrap"].use_positive_direction = False
            bpy.context.object.modifiers["Shrinkwrap"].use_negative_direction = True

            bpy.ops.object.modifier_apply(modifier="Shrinkwrap")
            frame = frame + 1
            print(frame, "UpJaw")
        bpy.ops.object.select_all(action="DESELECT")
        UpJawMoved = bpy.data.objects["UpJawMoved"]
        UpJawMoved.select_set(True)
        bpy.ops.object.delete()
        bpy.ops.object.select_all(action="DESELECT")
        UpJawDriver = bpy.data.objects["UpJawDriver"]
        UpJawDriver.select_set(True)
        bpy.ops.object.delete()

    #########################################################################################################

    def bakeLowPlane(self):
        print("low")

        for o in ("LowJaw", "UpJaw", "Occlusal_Plane"):
            obj = bpy.context.scene.objects.get(o)
            # if obj: obj.hide_viewport = False
            if obj:
                obj.hide_set(False)

        bpy.ops.object.select_all(action="DESELECT")
        Occlusal_Plane = bpy.data.objects["Occlusal_Plane"]
        Occlusal_Plane.select_set(True)

        bpy.context.view_layer.objects.active = Occlusal_Plane

        bpy.ops.object.duplicate()
        LowOcclPlane = bpy.context.object
        LowOcclPlane.select_set(True)
        LowOcclPlane.name = "LowOcclPlane"

        loc = LowOcclPlane.location
        (x, y, z) = (0.0, 0.0, -5.0)
        LowOcclPlane.location = loc + mathutils.Vector((x, y, z))

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.subdivide(number_cuts=6)
        bpy.ops.object.mode_set(mode="OBJECT")

        #########################################################
        # Shrinkwrap
        #########################################################

        object = bpy.data.objects["LowJaw"]
        scn = bpy.context.scene
        frame_start = scn.frame_start
        frame_end = scn.frame_end

        frame = frame_start

        while frame <= frame_end:
            bpy.context.scene.frame_set(frame)
            bpy.ops.object.modifier_add(type="SHRINKWRAP")
            bpy.context.object.modifiers["Shrinkwrap"].target = object
            bpy.context.object.modifiers["Shrinkwrap"].wrap_method = "PROJECT"
            bpy.context.object.modifiers["Shrinkwrap"].wrap_mode = "OUTSIDE"
            bpy.context.object.modifiers["Shrinkwrap"].use_project_z = True

            bpy.ops.object.modifier_apply(modifier="Shrinkwrap")
            frame = frame + 1
            print(frame)


#################################################################################################
# Registration :
#################################################################################################

classes = [
    BDENTAL_4D_OT_BakePlane,
    BDENTAL_4D_OT_LowJawChild,
]


def register():

    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
