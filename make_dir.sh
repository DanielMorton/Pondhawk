#!/bin/zsh

while :;
do
    case $1 in
        -d|--dir)
            DIR="$2"
            shift
            ;;
        *)  break
    esac
    shift
done

mkdir $DIR
mkdir $DIR/images

