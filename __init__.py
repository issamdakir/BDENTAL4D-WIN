# ----------------------------------------------------------
# File __init__.py
# ----------------------------------------------------------

#    Addon info
# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####
##############################################################################################
bl_info = {
    "name": "BDENTAL-4D-Win",  ###################Addon name
    "author": "Essaid Issam Dakir DMD, Ilya Fomenco DMD",
    "version": (1, 0, 0),
    "blender": (2, 90, 1),  ################# Blender working version
    "location": "3D View -> UI SIDE PANEL ",
    "description": "3D Tools suite for Digital Dentistry",  ########### Addon description
    "warning": "",
    "doc_url": "",
    "tracker_url": "",
    "category": "Dental",  ################## Addon category
}
#############################################################################################
# IMPORTS :
#############################################################################################
# Python imports :
import sys, os, bpy
from importlib import import_module
from os.path import dirname, join, realpath, abspath, exists


if sys.platform == "win32":
    sys.stdout.reconfigure(
        encoding="cp65001"
    )  # activate unicode characters in windows CLI

#############################################################
def ImportReq(REQ_DICT):
    Pkgs = []
    for mod, pkg in REQ_DICT.items():
        try:
            import_module(mod)
        except ImportError:
            Pkgs.append(pkg)

    return Pkgs


###################################################
REQ_DICT = {
    "SimpleITK": "SimpleITK==2.0.2",
    "vtk": "vtk==9.0.1",
    "cv2.aruco": "opencv-contrib-python==4.4.0.46",
    "matplotlib": "matplotlib",
    "scipy": "scipy",
    "PyQt5": "PyQt5",
}
ADDON_DIR = dirname(abspath(__file__))
REQ_ZIP_DIR = join(ADDON_DIR, "Resources", "REQ_ZIP_DIR")
BDENTAL_4D_Modules_DIR = join(os.path.expanduser("~"),"BDENTAL_4D_Modules")

if not sys.path[0] == BDENTAL_4D_Modules_DIR:
    sys.path.insert(0, BDENTAL_4D_Modules_DIR)

NotFoundPkgs = ImportReq(REQ_DICT)

if NotFoundPkgs:
    print(NotFoundPkgs)
    ############################
    # Install Req Registration :
    ############################
    from .Operators import BDENTAL_4D_InstallReq

    def register():

        BDENTAL_4D_InstallReq.register()

    def unregister():

        BDENTAL_4D_InstallReq.unregister()

    if __name__ == "__main__":
        register()

else:
    ######################
    # Addon Registration :
    ######################

    # Addon modules imports :
    from . import BDENTAL_4D_Props, BDENTAL_4D_Panel
    from .Operators import (
        BDENTAL_4D_Operators,
        JTracker_Operators,
        JTracker_WAXUP_Operators,
    )

    addon_modules = [
        BDENTAL_4D_Props,
        BDENTAL_4D_Panel,
        BDENTAL_4D_Operators,
        JTracker_Operators,
        JTracker_WAXUP_Operators,
    ]
    init_classes = []

    def register():

        for module in addon_modules:
            module.register()
        for cl in init_classes:
            bpy.utils.register_class(cl)

    def unregister():
        for cl in init_classes:
            bpy.utils.unregister_class(cl)
        for module in reversed(addon_modules):
            module.unregister()

    if __name__ == "__main__":
        register()
