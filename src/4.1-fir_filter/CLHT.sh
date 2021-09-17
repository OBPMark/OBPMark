# Open CL Header tool
for fi in $(find . -type f -name "*.cl"); do
    basename=$(basename -- "$fi")
    filename="${basename%.*}"
    dir=$(dirname "$fi")
    # Override emplace file content
    echo "" > ${dir}/GEN_$filename.hcl
    # Iterate line by line detecting tokens
    while read s || [ -n "$s" ]; do
        if [[ $s != "" ]]; then
            if [[ $s == "#htvar "* ]]; then
                # Add variable when token is found
                echo "std::string "${s#"#htvar "}" = " >> ${dir}/GEN_$filename.hcl
            elif [[ $s == "#htdefine "* ]]; then
                # Add macro when token is found
                echo "#define "${s#"#htdefine "}"" >> ${dir}/GEN_$filename.hcl
            elif [[ $s == "#htifdef "* ]]; then
                # Add macro when token is found
                echo "#ifdef "${s#"#htifdef "}"" >> ${dir}/GEN_$filename.hcl
            elif [[ $s == "#htifndef "* ]]; then
                # Add macro when token is found
                echo "#ifndef "${s#"#htifndef "}"" >> ${dir}/GEN_$filename.hcl
            elif [[ $s == "#htelse"* ]]; then
                # Add macro when token is found
                echo "#else" >> ${dir}/GEN_$filename.hcl
            elif [[ $s == "#htendif"* ]]; then
                # Add macro when token is found
                echo "#endif" >> ${dir}/GEN_$filename.hcl
            elif [[ $s == "#htendvar"* ]]; then
                # Add macro when token is found
                echo ";" >> ${dir}/GEN_$filename.hcl
            else 
                echo "$s" | sed 's/^.\{1,\}$/"&\\n"/' >> ${dir}/GEN_$filename.hcl
            fi
        fi  
    done < $fi
done
