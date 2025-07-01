for file in *; do
    if [ -f "$file" ]; then
        echo "Converting $file from cp949 to utf-8..."
        iconv -f cp949 -t utf-8 "$file" > "$file.tmp" && mv "$file.tmp" "$file"
        if [ $? -eq 0 ]; then
            echo "Successfully converted $file"
        else
            echo "Failed to convert $file"
        fi
    fi
done
