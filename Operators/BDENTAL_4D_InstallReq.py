# Python imports :
import os, bpy, shutil
from os import listdir
from os.path import dirname, join, realpath, abspath, exists, expanduser

import gpu
from gpu_extras.batch import batch_for_shader
import bgl
import blf

#############################################################
def ShowMessageBox(message=[], title="INFO", icon="INFO"):
    def draw(self, context):
        for txtLine in message:
            self.layout.label(text=txtLine)

    bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)


#############################################################
def ReqInstall(REQ_ZIP_DIR, BDENTAL_4D_Modules_DIR):

    if listdir(REQ_ZIP_DIR):
        ZipReqFiles = [join(REQ_ZIP_DIR, f) for f in listdir(REQ_ZIP_DIR)]
        for ZipFile in ZipReqFiles:
            shutil.unpack_archive(ZipFile, BDENTAL_4D_Modules_DIR)

        print("Requirements installed from ARCHIVE!")
        print("Please Restart Blender")
        message = [
            "Required Modules installation completed! ",
            "Please Restart Blender",
        ]
        ShowMessageBox(message=message, icon="COLORSET_03_VEC")

    else:

        message = [
            "Sorry can't find Requirement Zip File",
            "Please Uninstall the Addon . ",
            "Re-download the Addon from the official link",
            "Re-install the Addon !",
        ]
        for line in message:
            print(line)
        ShowMessageBox(message=message, icon="COLORSET_02_VEC")
        print(message)


#############################################################
# Install Requirements Operators :
#############################################################


class BDENTAL_4D_OT_InstallRequirements(bpy.types.Operator):
    """ Requirement installer """

    bl_idname = "bdental.installreq"
    bl_label = "INSTALL BDENTAL_4D MODULES"

    def execute(self, context):

        ADDON_DIR = dirname(dirname(abspath(__file__)))
        REQ_ZIP_DIR = join(ADDON_DIR, "Resources", "REQ_ZIP_DIR")
        BDENTAL_4D_Modules_DIR = join(expanduser("~/BDENTAL_4D_Modules_291"))

        if not exists(BDENTAL_4D_Modules_DIR) :
            os.mkdir(BDENTAL_4D_Modules_DIR)
            # shutil.rmtree(BDENTAL_4D_Modules_DIR)
        
        
        ReqInstall(REQ_ZIP_DIR, BDENTAL_4D_Modules_DIR)

        # if exists(BDENTAL_4D_Theme):
        #     bpy.ops.preferences.theme_install(filepath=BDENTAL_4D_Theme)

        return {"FINISHED"}


class BDENTAL_4D_PT_InstallReqPanel(bpy.types.Panel):
    """ Install Req Panel"""

    bl_idname = "BDENTAL_4D_PT_InstallReqPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"  # blender 2.7 and lower = TOOLS
    bl_category = "BDENTAL_4D"
    bl_label = "BDENTAL_4D"
    # bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.operator("bdental.installreq")


#################################################################################################
# Registration :
#################################################################################################

classes = [
    BDENTAL_4D_OT_InstallRequirements,
    BDENTAL_4D_PT_InstallReqPanel,
]


def register():

    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
