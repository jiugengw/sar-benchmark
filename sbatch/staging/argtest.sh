set -- "$@"; NAME="$1"; RUN="$2"; STAGES="${3:-plain,speckle}"; shift $(( $# < 3 ? $# : 3 )); echo "NAME=$NAME RUN=$RUN STAGES=$STAGES EXTRA=[$*]"
