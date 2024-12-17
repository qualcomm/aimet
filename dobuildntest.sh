#!/bin/bash
# =============================================================================
#  @@-COPYRIGHT-START-@@
#  
#  Copyright (c) 2020, Qualcomm Innovation Center, Inc. All rights reserved.
#  
#  Redistribution and use in source and binary forms, with or without 
#  modification, are permitted provided that the following conditions are met:
#  
#  1. Redistributions of source code must retain the above copyright notice, 
#     this list of conditions and the following disclaimer.
#  
#  2. Redistributions in binary form must reproduce the above copyright notice, 
#     this list of conditions and the following disclaimer in the documentation 
#     and/or other materials provided with the distribution.
#  
#  3. Neither the name of the copyright holder nor the names of its contributors 
#     may be used to endorse or promote products derived from this software 
#     without specific prior written permission.
#  
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#  
#  SPDX-License-Identifier: BSD-3-Clause
#  
#  @@-COPYRIGHT-END-@@
# =============================================================================

###############################################################################
## This is a script to build and run tests AIMET code.
## This script must be run from within the AIMET top-level folder
###############################################################################

# verbose mode
# set -x

# enable exit on error.
set -e

run_prep=1
run_clean=0
run_build=0
run_package_gen=0
run_unit_tests=0
run_code_violation=0
run_code_coverage=0
run_static_analysis=0
run_acceptance_tests=0

EXIT_CODE=0

variant_docs="tf-torch-cpu"

# Array to store python source file paths
declare -a PYTHON_SRC_PATHS=()
# Variable to store python source file paths
PYTHON_SRC_MODULE_PATHS=""
# PYTHONPATH variable
PYTHONPATH_VALUE=""

# Array to store python source file paths for code coverage
declare -a PYCOV_SRC_PATHS=()
# Array to store python test file paths for code coverage
declare -a PYCOV_TEST_PATHS=()

workspaceFolder=`pwd`
outputFolder=

function pre_exit {
    # Capture the exit code
    EXIT_CODE=$?

    if [ -z "$outputFolder" ]; then
        outputFolder=$workspaceFolder/buildntest_results
    fi

    summaryFile=${outputFolder}/summary.txt

    if [[ -f ${summaryFile} ]]; then
        # In case there is non-zero exit code, then add a FAILED tag to the summary file.
        if [ $EXIT_CODE -ne 0 ]; then
            echo -e "One or more Stages \t\t FAILED " | tee -a ${outputFolder}/summary.txt
        fi

        echo -e "----------------------------------------------------------------------------------------------------------\n" |tee -a ${summaryFile}
        echo -e "\nResults are in location:\n${outputFolder}\n" | tee -a ${summaryFile}
        cat ${summaryFile}

        if grep -q FAIL "${summaryFile}"; then
            EXIT_CODE=3
        fi
    fi

    # Return the exit code
    exit ${EXIT_CODE}
}
trap pre_exit EXIT

function check_stage() {
    RESULT=$1
    STAGE=$2
    if [ "$3" ]; then
        EXIT_ON_FAIL=$3
    fi

    if [ $RESULT -eq 0 ]; then
        echo -e "Stage $STAGE \t\t PASS " | tee -a ${outputFolder}/summary.txt
    else
        echo -e "Stage $STAGE \t\t FAILED " | tee -a ${outputFolder}/summary.txt
        if [ $EXIT_ON_FAIL == "true" ]; then
            echo -e "\n ABORTED " | tee -a ${outputFolder}/summary.txt
            exit 3
        fi
    fi
}

usage() {
  echo -e "\nThis is a script to build and run tests on AIMET code."
  echo -e "This script must be executed from within the AIMET repo's top-level folder."
  echo -e "NOTE: This script must be executed within the docker container (or in a machine with all dependencies installed). It will NOT start a docker container.\n"
  echo "${0} [-o <output_folder>]"
  echo "    -b --> build the code"
  echo "    -p --> generate pip packages"
  echo "    -u --> run unit tests"
  echo "    -v --> run code violation checks (using pylint tool)"
  echo "    -g --> run code coverage checks (using pycov tool)"
  echo "    -s --> run static analysis (using clang-tidy tool)"
  echo "    -a --> run acceptance tests (Warning: This will take a long time to complete!)"  
  echo "    -o --> optional output folder. Default is current directory"
  echo "    -w --> path to AIMET workspace. Default is current directory"
}

while getopts "o:w:abcghpsuv" opt;
   do
      case $opt in
         a)
             run_acceptance_tests=1
             ;;
         b)
             run_build=1
             ;;
         c)
             run_clean=1
             ;;
         g)
             run_code_coverage=1
             ;;
         p)
             run_package_gen=1
             ;;
         u)
             run_unit_tests=1
             ;;
         v)
             run_code_violation=1
             ;;
         s)
             run_static_analysis=1
             ;;
         h)
             usage
             exit 0
             ;;
         o)
             outputFolder=$OPTARG
             ;;
         w)
             workspaceFolder=$OPTARG
             ;;
         :)
             echo "Option -$OPTARG requires an argument" >&2
             exit 1;;
         ?)
             echo "Unknown arg $opt"
             usage
             exit 1
             ;;
      esac
done

# If no modes are enabled by user, then enable most modes by default
if [ $run_clean -eq 0 ] && [ $run_acceptance_tests -eq 0 ] && [ $run_build -eq 0 ] && \
    [ $run_package_gen -eq 0 ] && [ $run_unit_tests -eq 0 ] && [ $run_code_violation -eq 0 ] && \
    [ $run_code_coverage -eq 0 ] && [ $run_static_analysis -eq 0 ]; then
    run_prep=1
    run_clean=1
    run_build=1
    run_package_gen=1
    run_unit_tests=1
    run_code_violation=1
    run_code_coverage=1
    run_static_analysis=1
    run_acceptance_tests=0
fi

if [[ -z "${workspaceFolder}" ]]; then
    usage
    echo -e "ERROR: Workspace directory was not specified!"
    exit 3
fi

echo "Starting AIMET build and test..."
workspaceFolder=`readlink -f ${workspaceFolder}`
buildFolder=$workspaceFolder/build
installFolder=$workspaceFolder/build/staging/universal
artifactsFolder=$buildFolder/artifacts
AIMET_TORCH_HOME=${buildFolder}/torch_pretrain_data

# Sanity check to verify whether we're running form the correct repo location
if [[ ! -d "${workspaceFolder}/TrainingExtensions" ]] && [[ ! -d "${workspaceFolder}/aimet/TrainingExtensions" ]]; then
   echo -e "ERROR: Not in AIMET directory!"
   exit 3
fi

if [[ -d "${workspaceFolder}/Jenkins" ]]; then
    toolsFolder=${workspaceFolder}/Jenkins
elif [[ -d "${workspaceFolder}/aimet/Jenkins" ]]; then
    toolsFolder=${workspaceFolder}/aimet/Jenkins
fi

if [ $run_clean -eq 1 ]; then
    echo -e "\n********** Stage: Clean **********\n"
    if [ -d ${buildFolder} ]; then
        rm -rf ${buildFolder}/* | true
    fi
fi

if [ -z "$outputFolder" ]; then
    outputFolder=$buildFolder/results
fi
mkdir -p ${outputFolder}
if [ ! -f "${outputFolder}/summary.txt" ]; then
    touch ${outputFolder}/summary.txt
fi
if ! grep -q "AIMET Build and Test Summary" "${outputFolder}/summary.txt"; then
    echo -e "\n----------------------------------------------------------------------------------------------------------" | tee -a ${outputFolder}/summary.txt
    echo -e "\t\t AIMET Build and Test Summary " | tee -a ${outputFolder}/summary.txt
    echo -e "----------------------------------------------------------------------------------------------------------" | tee -a ${outputFolder}/summary.txt
fi

if [ $run_prep -eq 1 ]; then
    echo -e "\n********** Stage 1: Preparation **********\n"
    cd $workspaceFolder

    # Download the checkpoint files if they don't already exist
    #NOTE: We needed this due to some intermittant issues downloading via torchvision
    ## mkdir -p ${AIMET_TORCH_HOME}/checkpoints
    ## wget -N https://download.pytorch.org/models/resnet18-5c106cde.pth -P ${AIMET_TORCH_HOME}/checkpoints
    ## wget -N https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth -P ${AIMET_TORCH_HOME}/checkpoints

    github_url="https://github.com"
    if [[ ${GITHUB_MIRROR_URL} ]]; then
        github_url=${GITHUB_MIRROR_URL}
        export GIT_SSH_COMMAND='ssh -i /tmp/.ssh/id_rsa -o "StrictHostKeyChecking=no"'
    fi
    echo -e "Using ${github_url} for public repos..."

    # Clone the google test repo if not already present
    google_test_path="${workspaceFolder}/ThirdParty/google/googletest-release-1.12.1"
    if [ ! -e ${google_test_path} ]; then
        echo "Setting up googletest from internal sources"
        mkdir -p ${workspaceFolder}/ThirdParty/google
        pushd ${workspaceFolder}/ThirdParty/google
        if [[ -f "${DEPENDENCY_DATA_PATH}/googletest.zip" ]]; then
            unzip ${DEPENDENCY_DATA_PATH}/googletest.zip
        else
            echo "WARNING.! fetching googletest from external sources"
            git clone ${github_url}/google/googletest.git -b release-1.12.1 googletest-release-1.12.1
        fi
        popd
        check_stage $? "Preparation" "true"
    else
        echo "googletest package exists.. proceeding further"
    fi

    # Clone patchelf repo if not already present
    patch_elf_path="${buildFolder}/_deps/patchelf-src/bin/patchelf"
    if [ ! -e ${patch_elf_path} ]; then
        echo "Setting up patchelf from Internal sources"
        mkdir -p ${buildFolder}/_deps/patchelf-src/
        pushd ${buildFolder}/_deps/patchelf-src/
        if [[ -f "${DEPENDENCY_DATA_PATH}/patchelf.tar.gz" ]]; then
            tar -xvzf ${DEPENDENCY_DATA_PATH}/patchelf.tar.gz
        fi
        popd
        check_stage $? "Preparation" "true"
    fi

    # Array of python src file path endings
    declare -a python_src_path_endings=("TrainingExtensions/common/src/python/aimet_common")
    # TODO: the line below causes code violation failures in TrainingExtensions/tensorflow and TrainingExtensions/torch
    # python_src_path_endings+=("Examples/common")
    # Array of path endings of interest for code coverage and their corresponding test folders
    declare -a pycov_dir_endings=("TrainingExtensions/common/src/python:TrainingExtensions/common/test")

    if [ -n "$AIMET_VARIANT" ]; then
        # Add tensorflow and/or torch paths based on the variant
        if [[ "$AIMET_VARIANT" == *"tf"* ]]; then
            python_src_path_endings+=("TrainingExtensions/tensorflow/src/python/aimet_tensorflow")
            pycov_dir_endings+=("TrainingExtensions/tensorflow/src/python:TrainingExtensions/tensorflow/test")
            python_src_path_endings+=("Examples/tensorflow/compression")
            python_src_path_endings+=("Examples/tensorflow/quantization")
            python_src_path_endings+=("Examples/tensorflow/utils")
        fi
        if [[ "$AIMET_VARIANT" == *"torch"* ]]; then
            python_src_path_endings+=("TrainingExtensions/torch/src/python/aimet_torch")
            python_src_path_endings+=("Examples/torch/compression")
            python_src_path_endings+=("Examples/torch/quantization")
            python_src_path_endings+=("Examples/torch/utils")
            pycov_dir_endings+=("TrainingExtensions/torch/src/python:TrainingExtensions/torch/test")
        fi
        if [[ "$AIMET_VARIANT" == *"onnx"* ]]; then
            python_src_path_endings+=("TrainingExtensions/onnx/src/python/aimet_onnx")
            python_src_path_endings+=("Examples/onnx/quantization")
            python_src_path_endings+=("Examples/onnx/utils")
            pycov_dir_endings+=("TrainingExtensions/onnx/src/python:TrainingExtensions/onnx/test")
        fi
    else
        # For default variant, add both tensorflow and/or torch and/or onnx paths
        python_src_path_endings+=("TrainingExtensions/tensorflow/src/python/aimet_tensorflow")
        pycov_dir_endings+=("TrainingExtensions/tensorflow/src/python:TrainingExtensions/tensorflow/test")

        python_src_path_endings+=("TrainingExtensions/torch/src/python/aimet_torch")
        pycov_dir_endings+=("TrainingExtensions/torch/src/python:TrainingExtensions/torch/test")

        python_src_path_endings+=("TrainingExtensions/onnx/src/python/aimet_onnx")
        pycov_dir_endings+=("TrainingExtensions/onnx/src/python:TrainingExtensions/onnx/test")

        python_src_path_endings+=("Examples/torch/compression")
        python_src_path_endings+=("Examples/torch/quantization")
        python_src_path_endings+=("Examples/torch/utils")
        python_src_path_endings+=("Examples/tensorflow/compression")
        python_src_path_endings+=("Examples/tensorflow/quantization")
        python_src_path_endings+=("Examples/tensorflow/utils")
        python_src_path_endings+=("Examples/onnx/quantization")
        python_src_path_endings+=("Examples/onnx/utils")
    fi

    # Populate an array of python src paths for use in later stages
    for python_src_path_ending in "${python_src_path_endings[@]}"; do
    	# Find all paths
        PYTHON_SRC_PATHS+=($(find . -path "*$python_src_path_ending" -exec readlink -f {} \;))
    done

    # Populate the PYTHONPATH env variable value for use in later stages
    PYTHONPATH_VALUE+=$artifactsFolder
    for python_src_path in "${PYTHON_SRC_PATHS[@]}"; do
	    # Append the parent of each python src path to PYTHONPATH (separated by colon)
        python_src_path_parent=$(readlink -f ${python_src_path}/..)
        PYTHONPATH_VALUE+=":"
        PYTHONPATH_VALUE+=${python_src_path_parent}
	    # Append the same path also to a string (separated by space)
        PYTHON_SRC_MODULE_PATHS+=python_src_path_parent
        PYTHON_SRC_MODULE_PATHS+=" "
    done

    # Find and add the "Examples" folders to the python path
    python_example_paths+=($(dirname $(find . -name "Examples" | grep -v "build")))
    for python_example_path in "${python_example_paths[@]}"; do
        PYTHONPATH_VALUE+=":"
        python_example_path_abs=$(readlink -f ${python_example_path})
        PYTHONPATH_VALUE+=${python_example_path_abs}
    done

    echo "PYTHONPATH value = $PYTHONPATH_VALUE"

    # Loop over the directory endings
    for pycov_dir_ending in "${pycov_dir_endings[@]}"; do
        pycov_src_path_ending=${pycov_dir_ending%%:*}
        pycov_test_path_ending=${pycov_dir_ending#*:}
        # Find all absolute src and test folders ending in the endings of interest
        PYCOV_SRC_PATHS+=($(find . -path "*$pycov_src_path_ending" -exec readlink -f {} \; | grep -v build))
        PYCOV_TEST_PATHS+=($(find . -path "*$pycov_test_path_ending" -exec readlink -f {} \; | grep -v build))
    done

    # Just display all the code coverage paths for debugging purposes
    for ((index=0;index<${#PYCOV_SRC_PATHS[@]};++index)); do
        pycov_src_path=${PYCOV_SRC_PATHS[index]}
        pycov_test_path=${PYCOV_TEST_PATHS[index]}
        echo -e "pycov_src_path = $pycov_src_path, pycov_test_path = $pycov_test_path"
    done
fi

if [ $run_build -eq 1 ]; then
    echo -e "\n********** Stage 2: Build **********\n"

    mkdir -p $buildFolder
    cd $buildFolder

    extra_opts=""

    if [ -n "$SW_VERSION" ]; then
        extra_opts+=" -DSW_VERSION=${SW_VERSION}"
    fi

    if [ -n "$AIMET_VARIANT" ]; then
        # Add build options based on variant
        if [[ "$AIMET_VARIANT" == *"gpu"* ]]; then
            extra_opts+=" -DENABLE_CUDA=ON"
        elif [[ "$AIMET_VARIANT" == *"cpu"* ]]; then
            extra_opts+=" -DENABLE_CUDA=OFF"
        fi
        if [[ "$AIMET_VARIANT" == "${variant_docs}" ]]; then
            # For doc variant, cmake "test" targets are NOT supported AND we need to enable all frameworks.
            extra_opts+=" -DENABLE_TESTS=OFF -DENABLE_TENSORFLOW=ON -DENABLE_TORCH=ON -DENABLE_ONNX=ON"
        elif [[ "$AIMET_VARIANT" == *"tf"* ]]; then
            extra_opts+=" -DENABLE_TENSORFLOW=ON -DENABLE_TORCH=OFF -DENABLE_ONNX=OFF"
        elif [[ "$AIMET_VARIANT" == *"torch"* ]]; then
            extra_opts+=" -DENABLE_TENSORFLOW=OFF -DENABLE_TORCH=ON -DENABLE_ONNX=OFF"
        elif [[ "$AIMET_VARIANT" == *"onnx"* ]]; then
            extra_opts+=" -DENABLE_TENSORFLOW=OFF -DENABLE_TORCH=OFF -DENABLE_ONNX=ON"
        else
            echo -e "ERROR: Invalid variant string!"
            exit 3
        fi
    fi
    # Do not exit on failure by default from this point forward
    set +e
    cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_INSTALL_PREFIX=${installFolder} ${extra_opts} ..

    make -j 8
    check_stage $? "Build" "true"

    if [[ "$AIMET_VARIANT" == "${variant_docs}" ]]; then
        echo -e "\n********** Stage 2a: Generate Docs **********\n"
        if [ -n "$DOC_TARGET" ]; then
            make ${DOC_TARGET}
        else
            make doc
        fi
        check_stage $? "Generate Doc" "true"
    fi
fi


# Skip all other stages for the docs variant
if [[ "$AIMET_VARIANT" != "${variant_docs}" ]]; then

    if [ $run_package_gen -eq 1 ]; then
        cd $buildFolder

        echo -e "\n********** Stage 2b: Install **********\n"
        make install
        check_stage $? "Install" "true"

        echo -e "\n********** Stage 2c: Package **********\n"
        make packageaimet
        check_stage $? "Package" "true"
    fi

    if [ $run_unit_tests -eq 1 ]; then
        echo -e "\n********** Stage 3: Unit tests **********\n"
        cd $buildFolder
        set +e
        unit_test_cmd=""
        if [[ -z ${DEPENDENCY_DATA_PATH} ]]; then
            echo -e "DEPENDENCY_DATA_PATH was NOT set"
        else
            echo -e "DEPENDENCY_DATA_PATH was set to ${DEPENDENCY_DATA_PATH}"
            unit_test_cmd+="export DEPENDENCY_DATA_PATH=${DEPENDENCY_DATA_PATH} && "
        fi
        unit_test_cmd+="export TORCH_HOME=${AIMET_TORCH_HOME} && ctest --verbose"
        eval " $unit_test_cmd"
        unit_test_rc=$?
        python ${toolsFolder}/unittesthelper.py ${workspaceFolder}
        check_stage $unit_test_rc "Unit tests" "false"
    fi

    if [ $run_code_violation -eq 1 ]; then
        echo -e "\n********** Stage 4: Code violation checks **********\n"
        cd $workspaceFolder
        pylint_results_dir=$outputFolder/code_violation_result
        mkdir -p ${pylint_results_dir}

        for python_src_path in "${PYTHON_SRC_PATHS[@]}"; do
            # Construct the pylint results file name
            # Remove the top-level path from the full path for brevity
            pylint_results_file_name=$(echo ${python_src_path#${workspaceFolder}/})
            # Replace forward slashes with underscores
            pylint_results_file_name=$(echo $pylint_results_file_name | sed -e 's/\//_/g')        
            # Append the suffix and extension
            pylint_results_file_name+="_pylint_results.out"

            PYTHONPATH=$PYTHONPATH_VALUE \
            PYLINTHOME=${buildFolder} \
            pylint --rcfile=${workspaceFolder}/.pylintrc -r n --msg-template='{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}' ${python_src_path} 2>&1 \
            | tee ${pylint_results_dir}/${pylint_results_file_name}
            code_violation_result=$?

            if [ $code_violation_result -eq 0 ]; then
                if grep -q "error:" ${pylint_results_dir}/${pylint_results_file_name}; then
                    echo -e "\n********** ${python_src_path} code violation analysis results START **********\n"
                    cat ${pylint_results_dir}/${pylint_results_file_name}
                    echo -e "\n********** ${python_src_path} code violation analysis results END **********\n"
                fi
            fi
            check_stage $code_violation_result "Code violation checks: ${python_src_path}" "false"
        done
    fi

    if [ $run_static_analysis -eq 1 ]; then
        echo -e "\n********** Stage 5: Static analysis **********\n"
        static_analysis_result=0
        clangtidy_results_dir=$outputFolder
        mkdir -p ${clangtidy_results_dir}
        #TODO: Do not fail from the static analysis command since there are many unresolved errors
        set +e
        cd $buildFolder; python3 /usr/bin/run-clang-tidy.py >| ${clangtidy_results_dir}/clang-tidy_results.out
        static_analysis_result=$?
        set -e
        # Check for errors in static analysis log file and if found, display the log.
        if grep -q "error:" "${clangtidy_results_dir}/clang-tidy_results.out"; then
            static_analysis_result=1
            echo -e "\n********** Static analysis results START **********\n"
            cat ${clangtidy_results_dir}/clang-tidy_results.out
            echo -e "\n********** Static analysis results END **********\n"
        fi
        check_stage $static_analysis_result "Static analysis" "false"
    fi

    if [ $run_acceptance_tests -eq 1 ]; then
        echo -e "\n********** Stage 6: Acceptance tests **********\n"
        cd $buildFolder
        set +e
        make AcceptanceTests
        acceptance_test_rc=$?
        python ${toolsFolder}/acceptancetesthelper.py ${workspaceFolder}
        check_stage $acceptance_test_rc "Acceptance tests" "false"
    fi

    if [ $run_code_coverage -eq 1 ]; then
        echo -e "\n********** Stage 7: Code coverage **********\n"
        set +e
        pycov_results_dir=$outputFolder/coverage_test_results

        # Loop over the code coverage paths
        for ((index=0;index<${#PYCOV_SRC_PATHS[@]};++index)); do
            pycov_src_path=${PYCOV_SRC_PATHS[index]}
            pycov_test_path=${PYCOV_TEST_PATHS[index]}

            # Verify that the directories exist
            if [ ! -d ${pycov_src_path} ] || [ ! -d ${pycov_test_path} ]; then
                echo -e "\n[ERROR] Code coverage directories do not exist\n"
                coverage_test_rc=1
            fi

            # Construct the code coverage results file name
            # Remove the top-level path from the full path for brevity
            pycov_results_file_name=$(echo ${pycov_src_path#${workspaceFolder}/})
            # Replace forward slashes with underscores
            pycov_results_file_name=$(echo $pycov_results_file_name | sed -e 's/\//_/g')
            # Append the suffix and extension
            pycov_results_file_name+="_code_coverage.xml"

            # Run the code coverage
            TORCH_HOME=${AIMET_TORCH_HOME} PYTHONPATH=$PYTHONPATH_VALUE py.test --cov=${pycov_src_path} ${pycov_test_path} --cov-report xml:${pycov_results_dir}/${pycov_results_file_name}
            coverage_test_rc=$?
            cp -a ${pycov_results_dir}/${pycov_results_file_name} $outputFolder
        done
        check_stage $coverage_test_rc "Code coverage" "false"
    fi

fi # end if [[ "$AIMET_VARIANT" != "${variant_docs}" ]]

echo -e "\n outputFolder = ${outputFolder}"
if grep -q FAIL "${outputFolder}/summary.txt"; then
    EXIT_CODE=3
fi

exit $EXIT_CODE
