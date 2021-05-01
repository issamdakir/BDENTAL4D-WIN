import bpy, os, sys
from os.path import join, dirname, exists, abspath
from .Operators.BDENTAL_4D_Utils import *


ADDON_DIR = dirname(abspath(__file__))
Addon_Version_Path = join(ADDON_DIR, "Resources", "BDENTAL_4D_Version.txt")
if exists(Addon_Version_Path):
    with open(Addon_Version_Path, "r") as rf:
        lines = rf.readlines()
        Addon_Version_Date = lines[0].split(";")[0]
else:
    Addon_Version_Date = "  "
# Selected icons :
red_icon = "COLORSET_01_VEC"
orange_icon = "COLORSET_02_VEC"
green_icon = "COLORSET_03_VEC"
blue_icon = "COLORSET_04_VEC"
violet_icon = "COLORSET_06_VEC"
yellow_icon = "COLORSET_09_VEC"
yellow_point = "KEYTYPE_KEYFRAME_VEC"
blue_point = "KEYTYPE_BREAKDOWN_VEC"

Wmin, Wmax = -400, 3000


class BDENTAL_4D_PT_MainPanel(bpy.types.Panel):
    """Main Panel"""

    bl_idname = "BDENTAL_4D_PT_MainPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"  # blender 2.7 and lower = TOOLS
    bl_category = "BDENTAL-4D"
    bl_label = "BDENTAL-4D"
    # bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):

        # Draw Addon UI :
        layout = self.layout

        box = layout.box()

        row = box.row()
        row.alert = True
        row.alignment = "CENTER"
        row.label(text=f"WINDOWS VERSION : {Addon_Version_Date}")

        row = box.row()
        row.alignment = "CENTER"
        row.operator("bdental4d.template", text="BDENTAL_4D THEME")
        row.operator("bdental4d.open_manual")


class BDENTAL_4D_PT_ScanPanel(bpy.types.Panel):
    """Scan Panel"""

    bl_idname = "BDENTAL_4D_PT_ScanPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"  # blender 2.7 and lower = TOOLS
    bl_category = "BDENTAL-4D"
    bl_label = "SCAN"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):

        BDENTAL_4D_Props = context.scene.BDENTAL_4D_Props
        GroupNodeName = BDENTAL_4D_Props.GroupNodeName
        VGS = bpy.data.node_groups.get(GroupNodeName)

        # Draw Addon UI :
        layout = self.layout

        row = layout.row()
        split = row.split()
        col = split.column()
        col.label(text="Project Directory :")
        col = split.column()
        col.prop(BDENTAL_4D_Props, "UserProjectDir", text="")

        if BDENTAL_4D_Props.UserProjectDir:

            row = layout.row()
            split = row.split()
            col = split.column()
            col.label(text="Scan Data Type :")
            col = split.column()
            col.prop(BDENTAL_4D_Props, "DataType", text="")

            if BDENTAL_4D_Props.DataType == "DICOM Series":

                row = layout.row()
                split = row.split()
                col = split.column()
                col.label(text="DICOM Directory :")
                col = split.column()
                col.prop(BDENTAL_4D_Props, "UserDcmDir", text="")

                if BDENTAL_4D_Props.UserDcmDir:

                    Box = layout.box()
                    # Box.alert = True
                    row = Box.row()
                    row.alignment = "CENTER"
                    row.scale_y = 2
                    row.operator("bdental4d.volume_render", icon="IMPORT")
            if BDENTAL_4D_Props.DataType == "3D Image File":

                row = layout.row()
                split = row.split()
                col = split.column()
                col.label(text="3D Image File :")
                col = split.column()
                col.prop(BDENTAL_4D_Props, "UserImageFile", text="")

                if BDENTAL_4D_Props.UserImageFile:

                    Box = layout.box()
                    # Box.alert = True
                    row = Box.row()
                    row.alignment = "CENTER"
                    row.scale_y = 2
                    row.operator("bdental4d.volume_render", icon="IMPORT")
        if context.object:
            if context.object.name.startswith("BD") and context.object.name.endswith(
                "CTVolume"
            ):
                row = layout.row()
                row.operator("bdental4d.reset_ctvolume_position")
                row = layout.row()
                row.label(text=f"Threshold {Wmin} to {Wmax} HU :")
                row = layout.row()
                row.prop(BDENTAL_4D_Props, "Treshold", text="TRESHOLD", slider=True)

                layout.separator()

                row = layout.row()
                row.label(text="Segments :")

                Box = layout.box()
                row = Box.row()
                row.prop(BDENTAL_4D_Props, "SoftTreshold", text="Soft Tissu")
                row.prop(BDENTAL_4D_Props, "SoftSegmentColor", text="")
                row.prop(BDENTAL_4D_Props, "SoftBool", text="")
                row = Box.row()
                row.prop(BDENTAL_4D_Props, "BoneTreshold", text="Bone")
                row.prop(BDENTAL_4D_Props, "BoneSegmentColor", text="")
                row.prop(BDENTAL_4D_Props, "BoneBool", text="")

                row = Box.row()
                row.prop(BDENTAL_4D_Props, "TeethTreshold", text="Teeth")
                row.prop(BDENTAL_4D_Props, "TeethSegmentColor", text="")
                row.prop(BDENTAL_4D_Props, "TeethBool", text="")

                Box = layout.box()
                row = Box.row()
                row.operator("bdental4d.multitresh_segment")
            if context.object.name.startswith("BD") and context.object.name.endswith(
                ("CTVolume", "SEGMENTATION")
            ):
                row = Box.row()
                split = row.split()
                col = split.column()
                col.operator("bdental4d.addslices", icon="EMPTY_AXIS")
                col = split.column()
                col.operator("bdental4d.multiview")


class BDENTAL_4D_PT_MeshesTools_Panel(bpy.types.Panel):
    """ Model/Mesh Tools Panel"""

    bl_idname = "BDENTAL_4D_PT_MeshesTools_Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"  # blender 2.7 and lower = TOOLS
    bl_category = "BDENTAL-4D"
    bl_label = "MODEL/MESH TOOLS"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        BDENTAL_4D_Props = context.scene.BDENTAL_4D_Props
        layout = self.layout

        # Model color :
        layout.label(text="COLOR :", icon=yellow_point)
        Box = layout.box()
        row = Box.row()
        # row.alignment = "CENTER"
        row.operator("bdental4d.add_color", text="ADD COLOR", icon="MATERIAL")
        if context.object:
            mat = context.object.active_material
            if mat:
                row.prop(mat, "diffuse_color", text="")
            else:
                row.prop(BDENTAL_4D_Props, "no_material_prop", text="")

        row.operator("bdental4d.remove_color", text="REMOVE COLOR")

        # Join / Link ops :
        layout.separator()
        layout.label(text="OBJECT RELATION :", icon=yellow_point)
        Box = layout.box()
        row = Box.row()
        row.operator("bdental4d.parent_object", text="Parent", icon="LINKED")
        row.operator(
            "bdental4d.unparent_objects", text="Un-Parent", icon="LIBRARY_DATA_OVERRIDE"
        )
        row.operator("bdental4d.join_objects", text="Join", icon="SNAP_FACE")
        row.operator("bdental4d.separate_objects", text="Separate", icon="SNAP_VERTEX")

        # Model Repair Tools :
        layout.separator()
        layout.label(text="REPAIR TOOLS", icon=yellow_point)
        Box = layout.box()
        split = Box.split(factor=2 / 3, align=False)
        col = split.column()
        row = col.row(align=True)
        row.operator("bdental4d.decimate", text="DECIMATE", icon="MOD_DECIM")
        row.prop(BDENTAL_4D_Props, "decimate_ratio", text="")
        row = col.row()
        row.operator("bdental4d.fill", text="FILL", icon="OUTLINER_OB_LIGHTPROBE")
        row.operator(
            "bdental4d.retopo_smooth", text="RETOPO SMOOTH", icon="BRUSH_SMOOTH"
        )
        try:
            ActiveObject = bpy.context.view_layer.objects.active
            if ActiveObject:
                if ActiveObject.mode == "SCULPT":
                    row.operator(
                        "sculpt.sample_detail_size", text="", icon="EYEDROPPER"
                    )
        except Exception:
            pass

        col = split.column()
        row = col.row()
        # row.scale_y = 2
        row.operator("bdental4d.clean_mesh", text="CLEAN MESH", icon="BRUSH_DATA")
        row = col.row()
        row.operator("bdental4d.voxelremesh")

        # Cutting Tools :
        layout.row().separator()
        layout.label(text="CUT TOOLS", icon=yellow_point)
        Box = layout.box()
        row = Box.row()
        row.prop(BDENTAL_4D_Props, "Cutting_Tools_Types_Prop", text="")

        if BDENTAL_4D_Props.Cutting_Tools_Types_Prop == "Curve Cutter 1":
            row = Box.row()
            row.prop(BDENTAL_4D_Props, "CurveCutCloseMode", text="")
            row.operator(
                "bdental4d.curvecutteradd", text="ADD CUTTER", icon="GP_SELECT_STROKES"
            )
            row = Box.row()
            row.operator(
                "bdental4d.curvecuttercut", text="CUT", icon="GP_MULTIFRAME_EDITING"
            )

        elif BDENTAL_4D_Props.Cutting_Tools_Types_Prop == "Curve Cutter 2":
            row = Box.row()
            row.prop(BDENTAL_4D_Props, "CurveCutCloseMode", text="")
            row.operator(
                "bdental4d.curvecutteradd2", text="ADD CUTTER", icon="GP_SELECT_STROKES"
            )
            row = Box.row()
            row.operator(
                "bdental4d.curvecutter2_shortpath",
                text="CUT",
                icon="GP_MULTIFRAME_EDITING",
            )

        elif BDENTAL_4D_Props.Cutting_Tools_Types_Prop == "Square Cutter":

            # Cutting mode column :
            row = Box.row()
            row.label(text="Select Cutting Mode :")
            row.prop(BDENTAL_4D_Props, "cutting_mode", text="")

            row = Box.row()
            row.operator("bdental4d.square_cut", text="ADD CUTTER")
            row.operator("bdental4d.square_cut_confirm", text="CUT")
            row.operator("bdental4d.square_cut_exit", text="EXIT")

        elif BDENTAL_4D_Props.Cutting_Tools_Types_Prop == "Paint Cutter":

            row = Box.row()
            row.operator("bdental4d.paintarea_toggle", text="PAINT CUTTER")
            row.operator("bdental4d.paintarea_plus", text="", icon="ADD")
            row.operator("bdental4d.paintarea_minus", text="", icon="REMOVE")
            row = Box.row()
            row.operator("bdental4d.paint_cut", text="CUT")

        if context.active_object:
            if (
                "BDENTAL4D_Curve_Cut" in context.active_object.name
                and context.active_object.type == "CURVE"
            ):

                obj = context.active_object
                row = Box.row()
                row.prop(obj.data, "extrude", text="Extrude")
                row.prop(obj.data, "offset", text="Offset")

        # Make BaseModel, survey, Blockout :
        layout.separator()
        layout.label(
            text="MODELS : [Base - Hollow - Survey - Blockout]", icon=yellow_point
        )
        Box = layout.box()
        row = Box.row()
        row.alignment = "CENTER"
        row.prop(BDENTAL_4D_Props, "BaseHeight")
        row.operator("bdental4d.model_base")
        row.operator("bdental4d.add_offset")
        row = Box.row()
        row.alignment = "CENTER"
        row.operator("bdental4d.survey")
        row.operator("bdental4d.block_model")


class BDENTAL_4D_PT_Guide(bpy.types.Panel):
    """ Guide Panel"""

    bl_idname = "BDENTAL_4D_PT_Guide"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"  # blender 2.7 and lower = TOOLS
    bl_category = "BDENTAL-4D"
    bl_label = "SURGICAL GUIDE"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        BDENTAL_4D_Props = context.scene.BDENTAL_4D_Props
        layout = self.layout
        Box = layout.box()
        row = Box.row()
        row.alignment = "CENTER"
        row.prop(BDENTAL_4D_Props, "TeethLibrary")
        row.operator("bdental4d.add_teeth")

        layout.separator()

        Box = layout.box()
        row = Box.row()
        row.alignment = "CENTER"
        row.operator("bdental4d.add_implant")

        Box = layout.box()
        row = Box.row()
        row.alignment = "CENTER"
        row.prop(BDENTAL_4D_Props, "SleeveDiameter")
        row.prop(BDENTAL_4D_Props, "SleeveHeight")
        row = Box.row()
        row.alignment = "CENTER"
        row.prop(BDENTAL_4D_Props, "HoleDiameter")
        row.prop(BDENTAL_4D_Props, "HoleOffset")

        row = Box.row()
        row.alignment = "CENTER"
        row.operator("bdental4d.add_implant_sleeve")

        Box = layout.box()
        row = Box.row()
        row.operator("bdental4d.add_tube")
        row.prop(BDENTAL_4D_Props, "TubeCloseMode", text="")

        if context.active_object:
            if (
                "BDENTAL4D_Tube" in context.active_object.name
                and context.active_object.type == "CURVE"
            ):
                obj = context.active_object
                row = Box.row()
                row.prop(obj.data, "bevel_depth", text="Radius")
                row.prop(obj.data, "extrude", text="Extrude")
                row.prop(obj.data, "offset", text="Offset")

        row = Box.row()
        row.operator("bdental4d.add_splint")

        row = Box.row()
        row.operator(
            "bdental4d.splintcutteradd",
            text="ADD SPLINT CUTTER",
            icon="GP_SELECT_STROKES",
        )
        row.prop(BDENTAL_4D_Props, "CurveCutCloseMode", text="")
        if context.active_object:
            if (
                "BDENTAL4D_Splint_Cut" in context.active_object.name
                and context.active_object.type == "CURVE"
            ):
                obj = context.active_object
                row = Box.row()
                row.prop(obj.data, "bevel_depth", text="Radius")
                row.prop(obj.data, "extrude", text="Extrude")
                row.prop(obj.data, "offset", text="Offset")

        row = Box.row()
        row.operator(
            "bdental4d.splintcuttercut", text="CUT", icon="GP_MULTIFRAME_EDITING"
        )


####################################################################
class BDENTAL_4D_PT_Align(bpy.types.Panel):
    """ALIGN Panel"""

    bl_idname = "BDENTAL_4D_PT_Main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"  # blender 2.7 and lower = TOOLS
    bl_category = "BDENTAL-4D"
    bl_label = "ALIGN"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        BDENTAL_4D_Props = context.scene.BDENTAL_4D_Props
        AlignModalState = BDENTAL_4D_Props.AlignModalState
        layout = self.layout

        # Align Tools :
        layout.separator()
        Box = layout.box()
        row = Box.row()
        row.label(text="Align Tools")
        row = Box.row()
        row.operator("bdental4d.align_to_front", text="ALIGN FRONT", icon="AXIS_FRONT")
        row.operator("bdental4d.to_center", text="TO CENTER", icon="SNAP_FACE_CENTER")
        row.operator(
            "bdental4d.center_cursor", text="Center Cursor", icon="PIVOT_CURSOR"
        )
        split = Box.split(factor=2 / 3, align=False)
        col = split.column()
        row = col.row()
        row.operator("bdental4d.occlusalplane", text="OCCLUSAL PLANE")
        col = split.column()
        row = col.row()
        row.alert = True
        row.operator("bdental4d.occlusalplaneinfo", text="INFO", icon="INFO")

        # Align Points and ICP :
        split = layout.split(factor=2 / 3, align=False)
        col = split.column()
        row = col.row()
        row.operator("bdental4d.alignpoints", text="ALIGN")
        col = split.column()
        row = col.row()
        row.alert = True
        row.operator("bdental4d.alignpointsinfo", text="INFO", icon="INFO")

        Condition_1 = len(bpy.context.selected_objects) != 2
        Condition_2 = bpy.context.selected_objects and not bpy.context.active_object
        Condition_3 = bpy.context.selected_objects and not (
            bpy.context.active_object in bpy.context.selected_objects
        )
        Condition_4 = not bpy.context.active_object in bpy.context.visible_objects

        Conditions = Condition_1 or Condition_2 or Condition_3 or Condition_4
        if AlignModalState:
            self.AlignLabels = "MODAL"
        else:
            if Conditions:
                self.AlignLabels = "INVALID"

            else:
                self.AlignLabels = "READY"

        #########################################

        if self.AlignLabels == "READY":
            TargetObjectName = context.active_object.name
            SourceObjectName = [
                obj
                for obj in bpy.context.selected_objects
                if not obj is bpy.context.active_object
            ][0].name

            box = layout.box()

            row = box.row()
            row.alert = True
            row.alignment = "CENTER"
            row.label(text="READY FOR ALIGNEMENT.")

            row = box.row()
            row.alignment = "CENTER"
            row.label(text=f"{SourceObjectName} will be aligned to, {TargetObjectName}")

        if self.AlignLabels == "INVALID" or self.AlignLabels == "NOTREADY":
            box = layout.box()
            row = box.row(align=True)
            row.alert = True
            row.alignment = "CENTER"
            row.label(text="STANDBY MODE", icon="ERROR")

        if self.AlignLabels == "MODAL":
            box = layout.box()
            row = box.row()
            row.alert = True
            row.alignment = "CENTER"
            row.label(text="WAITING FOR ALIGNEMENT...")


##############################################################################
# JawTracker
##############################################################################


class BDENTAL_4D_PT_JawTrack(bpy.types.Panel):
    """ JawTrack Panel """

    bl_idname = "BDENTAL_4D_PT_JawTrack"  # Not importatnt alwas the same as class name
    bl_label = " JAW-TRACK "  # this is the title (Top panel bare)
    bl_space_type = "VIEW_3D"  # always the same if you want side panel
    bl_region_type = "UI"  # always the same if you want side panel
    bl_category = "BDENTAL-4D"  # this is the vertical name in the side usualy the name of addon :)
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):

        BDENTAL_4D_Props = context.scene.BDENTAL_4D_Props

        yellow_point = "KEYTYPE_KEYFRAME_VEC"
        red_icon = "COLORSET_01_VEC"
        green_icon = "COLORSET_03_VEC"

        layout = self.layout
        ##################################
        # Camera Calibration UI :
        ##################################
        Box = layout.box()
        row = Box.row()
        split = row.split()
        col = split.column()
        col.label(text="JTrack Project Directory:")
        col = split.column()
        col.prop(BDENTAL_4D_Props, "JTrack_UserProjectDir", text="")

        ProjDir = AbsPath(BDENTAL_4D_Props.JTrack_UserProjectDir)
        CalibFile = AbsPath(
            join(BDENTAL_4D_Props.JTrack_UserProjectDir, "calibration.pckl")
        )
        if exists(ProjDir):

            if not exists(CalibFile):
                Box = layout.box()
                row = Box.row()
                row.alignment = "CENTER"
                row.label(text="CAMERA CALIBRATION :", icon=red_icon)

                row = Box.row()
                split = row.split()
                col = split.column()
                col.label(text="Calibration Images :")
                col = split.column()
                col.prop(BDENTAL_4D_Props, "CalibImages", text="")

                row = Box.row()
                split = row.split()
                col = split.column()
                col.label(text="Square length in meters :")
                col = split.column()
                col.label(text="Marker length in meters :")

                row = Box.row()
                split = row.split()
                col = split.column()
                col.prop(BDENTAL_4D_Props, "UserSquareLength", text="")
                col = split.column()
                col.prop(BDENTAL_4D_Props, "UserMarkerLength", text="")
                row = Box.row()
                row.operator("bdental4d.calibration")

            else:
                Box = layout.box()
                row = Box.row()
                row.alignment = "CENTER"
                row.label(text="Camera Calibration OK!", icon=green_icon)

                row = Box.row()
                split = row.split()
                col = split.column()
                col.label(text="Video-Track :")
                col = split.column()
                col.prop(BDENTAL_4D_Props, "TrackFile", text="")

                row = Box.row()
                split = row.split()
                col = split.column()
                col.label(text="Tracking type :")
                col = split.column()
                col.prop(BDENTAL_4D_Props, "TrackingType", text="")

                row = Box.row()
                row.operator("bdental4d.startrack")

        ##################################
        layout.separator()
        ##################################

        ##################################
        # DATA READ UI :
        ##################################
        Box = layout.box()
        row = Box.row()
        row.alignment = "CENTER"
        row.label(text="DATA READ :")

        row = Box.row()
        split = row.split()
        col = split.column()
        col.label(text="Tracking data file :")
        col = split.column()
        col.prop(BDENTAL_4D_Props, "TrackedData", text="")

        # row.prop(BDENTAL_4D_Props, "TrackedData", text="Tracked data file")
        row = Box.row()
        row.operator("bdental4d.setupjaw")
        if bpy.context.scene.objects.get("UpJaw") is not None:
            row.operator("bdental4d.setlowjaw")
        else:
            row.alert = True
            row.label(text="Set UpJaw First!")
        row = Box.row()
        row.operator("bdental4d.addboards")
        row = Box.row()
        row.operator("bdental4d.datareader")
        row = Box.row()
        if bpy.data.objects.get("LowMarker") is not None:
            row.operator("bdental4d.smoothkeyframes")
        else:
            row.alert = True
            row.alignment = "CENTER"
            row.label(text="LowMarker not found")
        row.operator
        row = Box.row()
        row.operator("bdental4d.drawpath")


class BDENTAL_4D_PT_Measurements(bpy.types.Panel):
    """ Measurements Panel"""

    bl_idname = "BDENTAL_4D_PT_Measurements"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"  # blender 2.7 and lower = TOOLS
    bl_category = "BDENTAL-4D"
    bl_label = "MEASUREMENTS"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        BDENTAL_4D_Props = context.scene.BDENTAL_4D_Props
        layout = self.layout
        Box = layout.box()
        row = Box.row()
        row.operator("bdental4d.add_markup_point")
        row.operator("bdental4d.add_reference_planes")
        if BDENTAL_4D_Props.ActiveOperator == "bdental4d.add_reference_planes":
            Box = layout.box()
            Message = [
                "Click to place cursor, <ENTER> to Add point",
                "Please Add Reference points in this order :",
                "- Nasion , Right Orbital, Right Porion, Left Orbital, Left Porion",
                "Click <ENTER> to Add Reference Planes",
            ]
            for line in Message:
                row = Box.row()
                row.alert = True
                row.label(text=line)
        row = Box.row()
        row.operator("bdental4d.ctvolume_orientation")


class BDENTAL_4D_PT_Waxup(bpy.types.Panel):
    """ WaxUp Panel"""

    bl_idname = "BDENTAL_4D_PT_Waxup"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"  # blender 2.7 and lower = TOOLS
    bl_category = "BDENTAL-4D"
    bl_label = "WAXUP"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        BDENTAL_4D_Props = context.scene.BDENTAL_4D_Props
        #        scn = bpy.context.scene
        #        Occlusal_Plane = bpy.data.objects.get["Occlusal_Plane"]
        #        if bpy.data.objects.get("ObjectName") is not None:
        layout = self.layout
        row = layout.row()
        row.operator(
            "bdental4d.lowjawchildtolowmarker",
            text="Start LowJaw Movings",
            icon="LIBRARY_DATA_INDIRECT",
        )

        layout = self.layout
        split = layout.split(factor=2 / 3, align=False)
        col = split.column()
        row = col.row()
        row.operator("bdental4d.occlusalplane", text="Occlusal Plane")
        col = split.column()
        row = col.row()
        #        row.alert = True
        row.operator("bdental4d.occlusalplaneinfo", text="INFO", icon="INFO")

        layout = self.layout
        row = layout.row()
        row.prop(BDENTAL_4D_Props, "BakeLowPlane", text="Bake Lower")
        row.prop(BDENTAL_4D_Props, "BakeUpPlane", text="Bake Upper")
        Occlusal_Plane = bpy.data.objects.get("Occlusal_Plane")
        UpJaw = bpy.data.objects.get("UpJaw")
        LowJaw = bpy.data.objects.get("LowJaw")

        if BDENTAL_4D_Props.BakeLowPlane or BDENTAL_4D_Props.BakeUpPlane:
            if Occlusal_Plane is not None and UpJaw is not None and LowJaw is not None:
                row = layout.row()
                row.operator("bdental4d.bakeplane", text="START", icon="ONIONSKIN_ON")

            elif Occlusal_Plane is None:
                row = layout.row()
                row.alert = True
                row.label(text="Occlusal plane is not detected!")
            elif UpJaw is None:
                row = layout.row()
                row.alert = True
                row.label(text="UpJaw is not detected!")
            elif LowJaw is None:
                row = layout.row()
                row.alert = True
                row.label(text="LowJaw is not detected!")

        if BDENTAL_4D_Props.BakeLowPlane == True:
            print("Low Enabled")
        else:
            print("Low Disabled")

        if BDENTAL_4D_Props.BakeUpPlane == True:
            print("Up Enabled")
        else:
            print("Up Disabled")


##################################################################################

# Registration :
#################################################################################################

classes = [
    BDENTAL_4D_PT_MainPanel,
    BDENTAL_4D_PT_ScanPanel,
    BDENTAL_4D_PT_MeshesTools_Panel,
    BDENTAL_4D_PT_Align,
    BDENTAL_4D_PT_Guide,
    BDENTAL_4D_PT_JawTrack,
    BDENTAL_4D_PT_Measurements,
    BDENTAL_4D_PT_Waxup,
]


def register():

    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


##########################################################
# TreshRamp = VGS.nodes.get("TresholdRamp")
# ColorPresetRamp = VGS.nodes.get("ColorPresetRamp")
# row = layout.row()
# row.label(
#     text=f"Volume Treshold ({BDENTAL_4D_Props.Wmin}/{BDENTAL_4D_Props.Wmax} HU) :"
# )
# row.template_color_ramp(
#     TreshRamp,
#     "color_ramp",
#     expand=True,
# )
# row = layout.row()
# row.prop(BDENTAL_4D_Props, "Axial_Loc", text="AXIAL Location :")
# row = layout.row()
# row.prop(BDENTAL_4D_Props, "Axial_Rot", text="AXIAL Rotation :")