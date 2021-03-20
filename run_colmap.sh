#!/bin/zsh

AFF=0
ANGLE=4 #Init min triangulation angle
MULT=1 #Use multiple camera models
while :;
do
    case $1 in
        -d|--dir)
            DIR="$2"
            shift
            ;;
	-a|--aff)
	   AFF="$2"
	   shift
	   ;;
	-g|--angle)
	   ANGLE="$2"
	   shift
	   ;;
         -m|-mult)
	   MULT="$2"
	   shift
	   ;;
        *)  break
    esac
    shift
done

rm $DIR/database.db
rm -r $DIR/sparse

FEATURE_EXTRACTOR="~/Downloads/COLMAP.app/Contents/MacOS/colmap feature_extractor \
   --database_path $DIR/database.db \
   --image_path $DIR/images \
   --ImageReader.single_camera 1 \
   --SiftExtraction.estimate_affine_shape $AFF \
   --SiftExtraction.domain_size_pooling $AFF"

MATCHER="~/Downloads/COLMAP.app/Contents/MacOS/colmap exhaustive_matcher \
   --database_path $DIR/database.db \
   --SiftMatching.guided_matching $AFF"

MAPPER="~/Downloads/COLMAP.app/Contents/MacOS/colmap mapper \
    --database_path $DIR/database.db \
    --image_path $DIR/images \
    --output_path $DIR/sparse \
    --Mapper.num_threads 16 \
    --Mapper.init_min_tri_angle $ANGLE \
    --Mapper.multiple_models $MULT \
    --Mapper.extract_colors 1"

echo $FEATURE_EXTRACTOR
echo $MATCHER
echo $MAPPER


eval $FEATURE_EXTRACTOR
eval $MATCHER
mkdir $DIR/sparse
eval $MAPPER

