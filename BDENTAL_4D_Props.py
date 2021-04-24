import bpy
from os.path import abspath

from bpy.props import (
    StringProperty,
    IntProperty,
    FloatProperty,
    EnumProperty,
    FloatVectorProperty,
    BoolProperty,
)


def TresholdUpdateFunction(self, context):
    BDENTAL_4D_Props = context.scene.BDENTAL_4D_Props
    GpShader = BDENTAL_4D_Props.GroupNodeName
    Treshold = BDENTAL_4D_Props.Treshold

    CtVolumeList = [
        obj
        for obj in bpy.context.scene.objects
        if obj.name.startswith("BD") and obj.name.endswith("_CTVolume")
    ]
    if context.object in CtVolumeList:
        Vol = context.object
        Preffix = Vol.name[:5]
        GpNode = bpy.data.node_groups.get(f"{Preffix}_{GpShader}")

        if GpShader == "VGS_Marcos_modified":
            Low_Treshold = GpNode.nodes["Low_Treshold"].outputs[0]
            Low_Treshold.default_value = Treshold
        if GpShader == "VGS_Dakir_01":
            DcmInfo = eval(BDENTAL_4D_Props.DcmInfo)
            Wmin = DcmInfo["Wmin"]
            Wmax = DcmInfo["Wmax"]
            treshramp = GpNode.nodes["TresholdRamp"].color_ramp.elements[0]
            value = (Treshold - Wmin) / (Wmax - Wmin)
            treshramp = value


def text_body_update(self, context):
    props = context.scene.ODC_modops_props
    if context.object:
        ob = context.object
        if ob.type == "FONT":
            mode = ob.mode
            bpy.ops.object.mode_set(mode="OBJECT")
            ob.data.body = props.text_body_prop

            # Check font options and apply them if toggled :
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.font.select_all()

            dict_font_options = {
                "BOLD": props.bold_toggle_prop,
                "ITALIC": props.italic_toggle_prop,
                "UNDERLINE": props.underline_toggle_prop,
            }
            for key, value in dict_font_options.items():
                if value == True:
                    bpy.ops.font.style_toggle(style=key)

            ob.name = ob.data.body
            bpy.ops.object.mode_set(mode=mode)


def text_bold_toggle(self, context):
    if context.object:
        ob = context.object
        if ob.type == "FONT":
            mode = ob.mode
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.font.select_all()
            bpy.ops.font.style_toggle(style="BOLD")
            bpy.ops.object.mode_set(mode=mode)


def text_italic_toggle(self, context):
    if context.object:
        ob = context.object
        if ob.type == "FONT":
            mode = ob.mode
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.font.select_all()
            bpy.ops.font.style_toggle(style="ITALIC")
            bpy.ops.object.mode_set(mode=mode)


def text_underline_toggle(self, context):
    if context.object:
        ob = context.object
        if ob.type == "FONT":
            mode = ob.mode
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.font.select_all()
            bpy.ops.font.style_toggle(style="UNDERLINE")
            bpy.ops.object.mode_set(mode=mode)


class BDENTAL_4D_Props(bpy.types.PropertyGroup):

    #####################
    #############################################################################################
    # CT_Scan props :
    #############################################################################################
    #####################

    UserProjectDir: StringProperty(
        name="Project Directory Path",
        default="",
        description="Project Directory Path",
        subtype="DIR_PATH",
    )

    #####################

    UserDcmDir: StringProperty(
        name="DICOM Path",
        default="",
        description="DICOM Directory Path",
        subtype="DIR_PATH",
    )

    UserImageFile: StringProperty(
        name="User 3D Image File Path",
        default="",
        description="User Image File Path",
        subtype="FILE_PATH",
    )

    #####################

    Data_Types = ["DICOM Series", "3D Image File", ""]
    items = []
    for i in range(len(Data_Types)):
        item = (str(Data_Types[i]), str(Data_Types[i]), str(""), int(i))
        items.append(item)

    DataType: EnumProperty(items=items, description="Data type", default="DICOM Series")

    #######################

    DcmInfo: StringProperty(
        name="(str) DicomInfo",
        default="{'Deffault': None}",
        description="Dicom series files list",
    )
    #######################

    PngDir: StringProperty(
        name="Png Directory",
        default="",
        description=" PNG files Sequence Directory Path",
    )
    #######################

    SlicesDir: StringProperty(
        name="Slices Directory",
        default="",
        description="Slices PNG files Directory Path",
    )
    #######################

    NrrdHuPath: StringProperty(
        name="NrrdHuPath",
        default="",
        description="Nrrd image3D file Path",
    )
    Nrrd255Path: StringProperty(
        name="Nrrd255Path",
        default="",
        description="Nrrd image3D file Path",
    )

    #######################

    IcpVidDict: StringProperty(
        name="IcpVidDict",
        default="None",
        description="ICP Vertices Pairs str(Dict)",
    )
    #######################

    Wmin: IntProperty()
    Wmax: IntProperty()

    #######################
    # SoftTissueMode = BoolProperty(description="SoftTissue Mode ", default=False)

    GroupNodeName: StringProperty(
        name="Group shader Name",
        default="",
        description="Group shader Name",
    )

    #######################

    Treshold: IntProperty(
        name="Treshold",
        description="Volume Treshold",
        default=600,
        min=-400,
        max=3000,
        soft_min=-400,
        soft_max=3000,
        step=1,
        update=TresholdUpdateFunction,
    )
    Progress_Bar: FloatProperty(
        name="Progress_Bar",
        description="Progress_Bar",
        subtype="PERCENTAGE",
        default=0.0,
        min=0.0,
        max=100.0,
        soft_min=0.0,
        soft_max=100.0,
        step=1,
        precision=1,
    )
    SoftTreshold: IntProperty(
        name="SOFT TISSU",
        description="Soft Tissu Treshold",
        default=-300,
        min=-400,
        max=3000,
        soft_min=-400,
        soft_max=3000,
        step=1,
    )
    BoneTreshold: IntProperty(
        name="BONE",
        description="Bone Treshold",
        default=600,
        min=-400,
        max=3000,
        soft_min=-400,
        soft_max=3000,
        step=1,
    )
    TeethTreshold: IntProperty(
        name="Teeth Treshold",
        description="Teeth Treshold",
        default=1400,
        min=-400,
        max=3000,
        soft_min=-400,
        soft_max=3000,
        step=1,
    )
    SoftBool: BoolProperty(description="", default=False)
    BoneBool: BoolProperty(description="", default=False)
    TeethBool: BoolProperty(description="", default=False)

    SoftSegmentColor: FloatVectorProperty(
        name="Soft Segmentation Color",
        description="Soft Color",
        default=[0.8, 0.46, 0.4, 1.0],  # [0.63, 0.37, 0.30, 1.0]
        soft_min=0.0,
        soft_max=1.0,
        size=4,
        subtype="COLOR",
    )
    BoneSegmentColor: FloatVectorProperty(
        name="Bone Segmentation Color",
        description="Bone Color",
        default=[0.44, 0.4, 0.5, 1.0],  # (0.8, 0.46, 0.4, 1.0),
        soft_min=0.0,
        soft_max=1.0,
        size=4,
        subtype="COLOR",
    )
    TeethSegmentColor: FloatVectorProperty(
        name="Teeth Segmentation Color",
        description="Teeth Color",
        default=[0.55, 0.645, 0.67, 1.000000],  # (0.8, 0.46, 0.4, 1.0),
        soft_min=0.0,
        soft_max=1.0,
        size=4,
        subtype="COLOR",
    )

    #######################

    CT_Loaded: BoolProperty(description="CT loaded ", default=False)
    CT_Rendered: BoolProperty(description="CT Rendered ", default=False)
    sceneUpdate: BoolProperty(description="scene update ", default=True)
    AlignModalState: BoolProperty(description="Align Modal state ", default=False)

    #######################
    ActiveOperator: StringProperty(
        name="Active Operator",
        default="None",
        description="Active_Operator",
    )
    SleeveDiameter: FloatProperty(
        name="Sleeve Diameter",
        description="Sleeve Diameter",
        default=5.0,
        min=0.0,
        max=100.0,
        soft_min=0.0,
        soft_max=100.0,
        step=1,
        precision=1,
    )
    SleeveHeight: FloatProperty(
        name="Sleeve Height",
        description="Sleeve Height",
        default=5.0,
        min=0.0,
        max=100.0,
        soft_min=0.0,
        soft_max=100.0,
        step=1,
        precision=1,
    )
    HoleDiameter: FloatProperty(
        name="Hole Diameter",
        description="Hole Diameter",
        default=2.0,
        min=0.0,
        max=100.0,
        soft_min=0.0,
        soft_max=100.0,
        step=1,
        precision=1,
    )
    HoleOffset: FloatProperty(
        name="Hole Offset",
        description="Sleeve Offset",
        default=0.1,
        min=0.0,
        max=100.0,
        soft_min=0.0,
        soft_max=100.0,
        step=1,
        precision=1,
    )

    #########################################################################################
    # Mesh Tools Props :
    #########################################################################################

    # Decimate ratio prop :
    #######################
    decimate_ratio: FloatProperty(
        description="Enter decimate ratio ", default=0.5, step=1, precision=2
    )
    #########################################################################################

    CurveCutterNameProp: StringProperty(
        name="Cutter Name",
        default="",
        description="Current Cutter Object Name",
    )

    #####################

    CuttingTargetNameProp: StringProperty(
        name="Cutting Target Name",
        default="",
        description="Current Cutting Target Object Name",
    )

    #####################

    Cutting_Tools_Types = [
        "Curve Cutter 1",
        "Curve Cutter 2",
        "Square Cutting Tool",
        "Paint Cutter",
    ]
    items = []
    for i in range(len(Cutting_Tools_Types)):
        item = (
            str(Cutting_Tools_Types[i]),
            str(Cutting_Tools_Types[i]),
            str(""),
            int(i),
        )
        items.append(item)

    Cutting_Tools_Types_Prop: EnumProperty(
        items=items, description="Select a cutting tool", default="Curve Cutter 1"
    )

    cutting_mode_list = ["Cut inner", "Keep inner"]
    items = []
    for i in range(len(cutting_mode_list)):
        item = (str(cutting_mode_list[i]), str(cutting_mode_list[i]), str(""), int(i))
        items.append(item)

    cutting_mode: EnumProperty(items=items, description="", default="Cut inner")


#################################################################################################
# Registration :
#################################################################################################

classes = [
    BDENTAL_4D_Props,
]


def register():

    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.BDENTAL_4D_Props = bpy.props.PointerProperty(type=BDENTAL_4D_Props)


def unregister():

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.BDENTAL_4D_Props


# props examples :

# Axial_Loc: FloatVectorProperty(
#     name="AXIAL location",
#     description="AXIAL location",
#     subtype="TRANSLATION",
#     update=AxialSliceUpdate,
# )
# Axial_Rot: FloatVectorProperty(
#     name="AXIAL Rotation",
#     description="AXIAL Rotation",
#     subtype="EULER",
#     update=AxialSliceUpdate,
# )
################################################
# Str_Prop_Search_1: StringProperty(
#     name="String Search Property 1",
#     default="",
#     description="Str_Prop_Search_1",
# )
# Float Props :
#########################################################################################

# F_Prop_1: FloatProperty(
#     description="Float Property 1 ",
#     default=0.0,
#     min=-200.0,
#     max=200.0,
#     step=1,
#     precision=1,
#     unit="NONE",
#     update=None,
#     get=None,
#     set=None,
# )
#########################################################################################
# # FloatVector Props :
#     ##############################################
#     FloatV_Prop_1: FloatVectorProperty(
#         name="FloatVectorProperty 1", description="FloatVectorProperty 1", size=3
#     )
#########################################################################################
