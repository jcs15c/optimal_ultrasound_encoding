"""
    Add all the root of the project to sys.path.
"""

from pathlib import Path
import sys

### add current project dirs to syspath: these statements will be run on an import
# note that the following variables will not corrupt the namespace of other files on an import unless "from add_project_tree import *" is issued
# we are in the root folder
level = 1
# project path for the current project
project_root_dir = Path(__file__).parents[level]
# the current project has a depth of 2, add all the project dirs recursively
sys.path.append(str(project_root_dir))
