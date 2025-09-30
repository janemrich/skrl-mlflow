#!/bin/bash

SUPPORTED_FRAMEWORKS=("torch" "jax" "warp")
PYTHON_INTERPRETER="python3.11"
PYTHON_VENV_DIR="_envs"


function print_help() {
    echo -e "\nUsage:  $0 [-c] [-f <framework>] -e <example>"
    echo -e ""
    echo -e "  -c: Delete virtual environments before running"
    echo -e "  -e <example>: Example folder to run (e.g. gymnasium)"
    echo -e "  -f <framework>: ML framework to use [ torch, jax, warp ]"
    echo -e ""
}

function python_virtual_environment() {
    local folder="$1"
    $PYTHON_INTERPRETER -m venv "$PYTHON_VENV_DIR/env_$folder"
    python_executable="$PYTHON_VENV_DIR/env_$folder/bin/python"
    python_executable=$($python_executable -c "import sys; print(sys.executable)")
    $python_executable -m pip install --quiet --upgrade pip
    echo "$python_executable"
}

function install_skrl() {
    local python_executable="$1"
    local frameworks=("${@:2}")
    $python_executable -m pip install --quiet -e ..
    for framework in "${frameworks[@]}"; do
        echo "  |     |-- Installing ML framework ($framework)..."
        $python_executable -m pip install --quiet -e ..[$framework]
    done
}

function run_scripts() {
    local example="$1"
    local python_executable="$2"
    local frameworks=("${@:3}")
    echo "  |-- Running '$example' scripts..."
    # change CWD to example folder
    cd "$example"
    # create result files
    echo "execution_code,duration,file" > result-stats.txt
    echo "" > result-errors.txt
    # create temporary files for stdout and stderr
    stdout_file=$(mktemp)
    stderr_file=$(mktemp)
    # run scripts
    for framework in "${frameworks[@]}"; do
        for file in ${framework}_*.py; do
            if [ -f "$file" ]; then
                echo "  |     |-- Running '$file'..."
                start=$(date +%s.%N)
                $python_executable $file --headless >"$stdout_file" 2>"$stderr_file"
                result=$?  # get execution code (success: 0)
                end=$(date +%s.%N)
                # log data (execution code, duration, file)
                duration=$(echo "$end - $start" | bc)
                echo "$result,$duration,$file" >> result-stats.txt
                # log errors, if any
                if [ $result -ne 0 ]; then
                    stdout=$(<"$stdout_file")
                    stderr=$(<"$stderr_file")
                    echo -e "\n$file" >> result-errors.txt
                    echo "$stderr" >> result-errors.txt
                fi
                # remove temporary files
                rm "$stdout_file" "$stderr_file"
            fi
        done
    done
    # change CWD back to parent folder
    cd ..
}


# parse optional arguments
examples=()
frameworks=("${SUPPORTED_FRAMEWORKS[@]}")
while getopts "cf:e:h" flag; do
    case "$flag" in
        c)
            echo "[Info] Deleting virtual environments..."
            rm -r "$PYTHON_VENV_DIR"
            mkdir -p "$PYTHON_VENV_DIR"
            ;;
        e)
            examples=($OPTARG)
            ;;
        f)
            frameworks=($OPTARG)
            ;;
        h)
            print_help
            exit 0
            ;;
    esac
done
shift $((OPTIND - 1))

# check given framework(s)
for framework in "${frameworks[@]}"; do
    if [[ ! " ${SUPPORTED_FRAMEWORKS[@]} " =~ " ${framework} " ]]; then
        echo "[Error] Framework '$framework' not supported. Supported frameworks: ${SUPPORTED_FRAMEWORKS[@]}"
        print_help
        exit 0
    fi
done

# check example folder(s)
if [ -z "$examples" ]; then
    echo "[Error] No example folder provided" >&2;
    print_help
    exit 0
fi

echo "[Info] Frameworks: ${frameworks[@]}"
echo "[Info] Examples: ${examples[@]}"

for example in "${examples[@]}"; do
    echo ""
    echo "[Info] Example: $example"
    # setup virtual environment
    echo "  |-- Creating virtual environment..."
    python_executable=$(python_virtual_environment "$example")
    echo "  |     |-- Python executable: $python_executable"
    echo "  |-- Installing skrl..."
    install_skrl "$python_executable" "${frameworks[@]}"
    # install per-example dependencies
    if [[ $example == "gym" ]]; then
        echo "  |-- Installing gym dependencies..."
        $python_executable -m pip install --quiet gym
    fi
    # run examples
    run_scripts "$example" "$python_executable" "${frameworks[@]}"
done
