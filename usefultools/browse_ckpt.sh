


<< EOF
Usage:    ./usefultools/browse_ckpt.sh   ~.ckpt

help  (暂没写文档)

ls state_dict

^C

EOF

shell_dict=$(dirname "$0")/../jscv/utils/shell_dict.py

python $shell_dict ${@:1}
