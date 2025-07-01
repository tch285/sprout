#!/bin/bash

# Get abspath of package
PACKAGE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "PACKAGE_DIR IS $PACKAGE_DIR"
PACKAGE_NAME=$(basename "${PACKAGE_DIR}")
echo "PACKAGE_NAME IS $PACKAGE_NAME"

# Default modulefile location or allow user to specify
MODULE_DIR="${1:-$PACKAGE_DIR/modules}"
echo "MODULE_DIR IS $MODULE_DIR"

mkdir -p "$MODULE_DIR/sprout"

cat > "${MODULE_DIR}/${PACKAGE_NAME}/0.1.0" << EOF
#%Module

proc ModulesHelp { } {
    # Help text for \`module help sprout\`
    puts stderr "Adds ${PACKAGE_NAME} to PYTHONPATH"
    puts stderr "\nThis module adds the Python package located at:"
    puts stderr "${PACKAGE_DIR}"
}

# Description for \`module whatis sprout\`
module-whatis "Python package: ${PACKAGE_NAME}"

set package_root "${PACKAGE_DIR}/src"

if { ![file exists \$package_root] } {
    puts stderr "Error: Package path does not exist: \$package_root"
    break
}

prepend-path PYTHONPATH \$package_root
# Add package directory to front of PYTHONPATH
EOF

echo "Created modulefile: ${MODULE_DIR}/${PACKAGE_NAME}"
echo "Add this to your MODULEPATH to use:"
echo "  module use ${MODULE_DIR}"