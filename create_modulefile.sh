#!/bin/bash

# Get the absolute path of the package directory
PACKAGE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "PACKAGE_DIR IS $PACKAGE_DIR"
PACKAGE_NAME=$(basename ${PACKAGE_DIR})
echo "PACKAGE_NAME IS $PACKAGE_NAME"
# basename gets the last part of the path
# If PACKAGE_DIR is /home/user/mypackage, PACKAGE_NAME becomes mypackage

# Default modulefile location or allow user to specify
MODULE_DIR="${1:-$PACKAGE_DIR/modules}"
echo "MODULE_DIR IS $MODULE_DIR"
# If user provides an argument ($1), use that
# Otherwise, default to $HOME/modulefiles

# Create modulefiles directory if it doesn't exist
mkdir -p "$MODULE_DIR"

# Create the modulefile using heredoc
cat > "${MODULE_DIR}/${PACKAGE_NAME}" << EOF
#%Module  # TCL modulefile header

proc ModulesHelp { } {
    # Help text shown when running: module help mypackage
    puts stderr "Adds ${PACKAGE_NAME} to PYTHONPATH"
    puts stderr "\nThis module adds the Python package located at:"
    puts stderr "${PACKAGE_DIR}"
}

module-whatis "Python package: ${PACKAGE_NAME}"
# Short description shown when running: module whatis mypackage

set package_root "${PACKAGE_DIR}"

if { ![file exists \$package_root] } {
    # TCL code to verify package directory exists
    puts stderr "Error: Package path does not exist: \$package_root"
    break
}

prepend-path PYTHONPATH \$package_root
# Add package directory to front of PYTHONPATH
EOF

# Inform user
echo "Created modulefile: ${MODULE_DIR}/${PACKAGE_NAME}"
echo "Add this to your MODULEPATH to use:"
echo "export MODULEPATH=\${MODULEPATH}:${MODULE_DIR}"