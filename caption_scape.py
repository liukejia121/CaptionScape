"""
Crop remote sensing image into tiles, then predict caption of each tile
and plot all the captions on the remote sensing image, call it caption scape.
In caption scape image, the captions will overlapping on the image in form of 
semitransparent rectangle, so that we can see the predicted caption of 
the tile and it's corresponding pixels at the same time. 
In this way, the effect of the SkyScript algorithm can be verified directly.
Related work is encapsulated into class CapMap.
"""

import os
import re
import csv
import math
import random
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from socket import gethostname
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
# Dynamic set the font to support the Chinese font, such as using SimHei.
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False 
from matplotlib.font_manager import FontProperties
from matplotlib.colors import hsv_to_rgb
# Import relative modules
from test_zero_shot_classification import test
from params import parse_args_cap
from classnames import CLASSNAMES_70, CLASSNAMES_70CN


class CapMap(object):
    """docstring for CapMap.
    Input:
        Remote sensing image (RSI),
        Initializtion parameters.
    operation:
        Crop remote sensing image into tiles, then predict caption of each tile
        and plot all the captions on the remote sensing image.
    Output:
        Mini dataset composed of RS image's tiles, with the same form of SkyScript_cls.
        Prediction result of each tile's class label, save to csv file.
        The caption scape image (CSI) of RS image with colored labels on it.
    """
    def __init__(self, args):
        super(CapMap, self).__init__()
        # Parse patrameters.
        args = parse_args_cap(args)
        
        self.dirf_pretrained_vit_model = args.pretrained_model

        # Suppose you have a pc and server to run this programe.
        DIR_ROOT_DSET1 = args.rootpath   # Your root dir of pc.
        DIR_ROOT_DSET2 = "/home/lkj/dataset"   # Your root dir of server.

        # Determine the host name currently running, and the appropriate root path is selected.
        self.hostname = gethostname()
        # The hostname of the current computer is: DESKTOP-NRBGBOO (change to your host name).
        if self.hostname == "DESKTOP-NRBGBOO" or self.hostname == 'p72':  
            self.dir_root_dset = DIR_ROOT_DSET1
        elif self.hostname == "vipa-01":  
            self.dir_root_dset = DIR_ROOT_DSET2  
        else:
            self.dir_root_dset = args.rootpath

        # The name of document containing your RS images.
        self.name_dset_src = args.imagesrc  
        # The prefix of your output dataset and result.
        self.imset = args.imset  
        # The dir of you prediction results and caption map image.
        self.name_save_result = args.savepath  #'rsitile_pred'.
        # The showing label language option. 
        self.label_language = args.label_language

        # Gets all the file names in the folder.
        lst_bigmap_name = os.listdir(self.dir_root_dset + '/' + self.name_dset_src)
        lst_bigmap_name = [x for x in lst_bigmap_name if ('.jpg' in x or '.png' in x or '.bmp' in x)]
        lst_bigmap_name = sorted(lst_bigmap_name)

        # Generate a list of all English letters (upper and lowercase), 52 elements.
        alphabet_list = [chr(i) for i in range(ord('a'), ord('z') + 1)] + \
                        [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        alphabet_list = sorted(alphabet_list)

        # Generates a dictionary of key value pairs: Data set name: corresponding RS image.
        self.lst_key_dsname = [f'{self.imset}{x}' for x in alphabet_list[:len(lst_bigmap_name)]]
        self.dct_dsn_bmn = dict(zip(self.lst_key_dsname, lst_bigmap_name))
        print("User input RS image dictionary:\n", self.dct_dsn_bmn)

        # With 52 letters, the maximum number can correspond to 52 large picture names.
        self.bigmap_index = args.x  
        self.tile_size = args.s           # side of tile.
        self.crop_size = args.s + args.c  # side of crop.
        # The name of the dataset generated with the remote sensing map you entered.
        self.name_dataset  = f'{self.imset}{args.x}{args.s}{args.s+args.c}'  # i.e. 'imsetA36'.
        # Get the current RS image name to process.
        self.name_xmap = self.dct_dsn_bmn[f'{self.imset}{args.x}']  

        # Get the "nickname" of the RS image.
        # i.e., '30.305,119.947', [30.305,119.947]  or '0,0', [0,0] .
        self.namy_xmap, lst_latlon = self._shorten_name_and_get_lonlat(self.name_xmap)  
        
        # Open RS image and get the size infomation.
        self.src_image = self._open_src_image()
        # unit pixels. height value in pixels.
        self.lat_pixel_xmap_span = self.src_image.size[1]  
        # unit pixels. width value in pixels.
        self.lon_pixel_xmap_span = self.src_image.size[0]  
        
        # Get or estimate the width of RS image in unit of meter.
        if lst_latlon[0]==0 and lst_latlon[1]==0:
            self.RSI_WIDTH_METER = 1000  # Hypothetical RS image lateral span value in meter.
            self.RSI_HEIGHT_METER = round(self.RSI_WIDTH_METER * self.lat_pixel_xmap_span / self.lon_pixel_xmap_span)
        else:
            self.RSI_WIDTH_METER = 1153  # Get the width value of RS image in meters.
            self.RSI_HEIGHT_METER = 700
        self.RSI_LON_METER_PER_PIXEL = self.RSI_WIDTH_METER/self.lon_pixel_xmap_span  
        self.RSI_LAT_METER_PER_PIXEL = self.RSI_HEIGHT_METER/self.lat_pixel_xmap_span

        # The lontitude span of RS image, tile and crop (in unit of degree)
        self.lon_angle_xmap_span = self.RSI_WIDTH_METER / (111000 * math.cos(lst_latlon[0] / 180 * math.pi))
        self.lon_angle_tile_span = self.tile_size * 0.0001  
        self.lon_angle_crop_span = self.crop_size * 0.0001  
        
        # The pixel span of tile and crop in unit of pixel.
        self.lon_pixel_tile_span = 111000 * self.lon_angle_tile_span / self.RSI_LON_METER_PER_PIXEL 
        self.lon_pixel_crop_span = 111000 * self.lon_angle_crop_span / self.RSI_LON_METER_PER_PIXEL 

        # The number of tile in RS image in horizental direction.
        self.grid_factor = self.lon_angle_xmap_span // self.lon_angle_tile_span   
                
        # Naming rule for the result file: dataset name-latitude longitude-tile span-crop span.
        namy_ = self.namy_xmap   # '30.305,119.947'
        xmap_ = f'{self.lon_angle_xmap_span*10000:.0f}'  # i.e., '0.0120'.
        tile_ = f'{self.lon_angle_tile_span*10000:.0f}'  # i.e., '0.0008'.
        crop_ = f'{self.lon_angle_crop_span*10000:.0f}'  # i.e., '0.0009'.
        self.name_result = f'{self.name_dataset}-cor{namy_}-span{xmap_}-tile{tile_}-crop{crop_}'

        # The relative csv file path to save the prediction result.
        self.dir_save_result = self.dir_root_dset + '/' + self.name_save_result
        self._dir_check_and_make(self.dir_save_result)
        # Get the dir of prediction result csv file.
        self.dirf_pred_csv = self.dir_save_result + '/' + self.name_result + '.csv' 

    def capmap_crop(self):
        """
        Crop RS image into a batch of tiles.
        Make up document tree as SkyScript_cls as follows:
            SkyScript_cls (Your mini dataset of RS image may has another name. )
                val (contains tiles of RS image)
                classnames.txt (70 class labels)
                img_txt_pairs_val.csv (3 columns, filepath-title-label infomation of tiles )
        """
        # The dir of your mini dataset of RSI 
        dir_dset = self.dir_root_dset + '/' + self.name_dataset  # "/home/lkj/dataset"/'arbm52a_cls'
        dir_dset_val = dir_dset + '/' + 'val'
        self._dir_check_and_make(dir_dset)
        self._dir_check_and_make(dir_dset_val)
        
        # Extract the central latitude and longtitude of RSI frome it's name
        _, lst_latlon = self._shorten_name_and_get_lonlat(self.name_xmap)  
        
        # Calculate the number of tiles
        n_tile = int(self.grid_factor // 2 + 5)
        # Based on the n_tile, generate a list of step increments in the longitude and latitude directions. 
        # Pay special attention to the fact that the change direction of the angle values of 
        # longitude and latitude may not be consistent with the pixel coordinate values of the image. 
        # First, consider the "Equator X--Prime Meridian Y coordinate system" in the first quadrant: 
        # latitude increases upwards, and longitude increases to the right.
        lst_index = [x for x in range(-n_tile, n_tile+1)] 
        # i.e., lst_index = [...,-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, ...]
        lst_lon = [lst_latlon[1] + x * self.lon_angle_tile_span for x in lst_index]
        # The image pixels are arranged from top to bottom vertically, 
        # while the latitude increases from bottom to top.
        lst_lat = [lst_latlon[0] + x * self.lon_angle_tile_span for x in lst_index]
        
        # Open RSI, crop and resize it based on the calculated coordinates.
        cnt = 1  # Count the number of tiles.
        lst_label_info = []  # Store label information.
        for j, lat_ in enumerate(lst_lat):      # row
            for i, lon_ in enumerate(lst_lon):  # col
                
                # First calculate the difference between the current tile's latitude and 
                # longitude coordinates and the center latitude and longitude coordinates 
                # of the large image. Pay special attention that in the first quadrant, 
                # the latitude increases from bottom to top.
                delta_lon_d, delta_lat_d = lon_-lst_latlon[1], lat_-lst_latlon[0]  
                # The larger the latitude coordinate, the smaller the vertical pixel coordinate.
                delta_lat_d = - delta_lat_d  
                
                # Convert the difference between the new coordinates and 
                # the center coordinates of the RSI: degrees to meters to pixels.
                delta_lon_m, delta_lat_m = self.angle2meter(delta_lon_d, delta_lat_d, lat_)
                delta_lon_p, delta_lat_p = self.meter2pixel(delta_lon_m, delta_lat_m)
                
                # Pixel difference: Adding to the center pixel coordinates of the RSI,
                # equals center pixel coordinates of the tile.
                lon_pix_tile_cnt = delta_lon_p + self.lon_pixel_xmap_span // 2
                lat_pix_tile_cnt = delta_lat_p + self.lat_pixel_xmap_span // 2
                
                # Tiles' central pixel coordinate.
                left  = lon_pix_tile_cnt - self.lon_pixel_crop_span//2
                right = lon_pix_tile_cnt + self.lon_pixel_crop_span//2
                up    = lat_pix_tile_cnt - self.lon_pixel_crop_span//2
                down  = lat_pix_tile_cnt + self.lon_pixel_crop_span//2

                # Tiles' central pixel coordinate overboundary detection.
                if not (left<0 or right>self.lon_pixel_xmap_span or up<0 or down>self.lat_pixel_xmap_span ) :  
                    # tiles'index corrdination：tile located in the row j and col i of RSI.
                    cor_index = f"{i:03.0f}" + "-" + f"{j:03.0f}"
                    # The comparative dir of csv.
                    filepath = (self.name_dataset  
                                + "/" + "val"  
                                + "/" + str(cnt)  
                                + "_" + cor_index  
                                + "_" + str(f"{lat_:.7f}")
                                + "," + str(f"{lon_:.7f}") + ".jpg")
                    # Cropping tile (left,down,right,up).
                    tile = self.src_image.crop((left, up, right, down))
                    # Processing tile.
                    tile = np.array(tile)
                    tile = Image.fromarray(tile)
                    # The absolute dir of csv.
                    path = (self.dir_root_dset + "/" + filepath)
                    #  Saving tile.
                    tile.save(path)
                    
                    # Recorde the infomation of filepath,title,label.
                    templabel = str(cnt % 69)  #'0'
                    cnt += 1
                    lst_label_info.append([filepath, 'a satellite image of annual crop', templabel])
            
        # Write list to csv.
        csvfilename = 'img_txt_pairs_val.csv'
        self._write_list_to_csv_in_patha(lst_label_info, csvfilename, dir_dset)
        
        # Get classnames.txt and write it to your mini dataset.
        lst_file = ['classnames.txt', 'classnames_70cn.txt']
        # lst_file = ['classnames.txt']
        self._copy_files_from_pathA2B(lst_file, self.dir_src_bigmaps, dir_dset)
    
    def capmap_pred(self):
        """
        Inferenc the label of each tile, save predictions to csv file.
        """
        # The dir of checkpoit file
        PRETRAINED_VIT_MODEL = self.dirf_pretrained_vit_model
        
        # Update the dictionary infomation, add your mini dataset info.  
        from benchmark_dataset_info import BENCHMARK_DATASET_ROOT_DIR
        from benchmark_dataset_info import BENCHMARK_DATASET_INFOMATION
        BENCHMARK_DATASET_ROOT_DIR = self.dir_root_dset  
        # Here we use 'SkyScript_cls' to replace self.name_dataset,
        # but the content of csv file is linked to your mini dataset.
        BENCHMARK_DATASET_INFOMATION['SkyScript_cls'] = {
            'classification_mode': 'multiclass',
            'test_data': BENCHMARK_DATASET_ROOT_DIR + f'/{self.name_dataset}/img_txt_pairs_val.csv',
            'classnames': BENCHMARK_DATASET_ROOT_DIR + f'/{self.name_dataset}/classnames.txt',
            'csv_separator': ',',
            'csv_img_key': 'filepath',
            'csv_class_key': 'label',
        }
        
        # Prepare csv file and classnames.txt file.
        test_data = BENCHMARK_DATASET_INFOMATION['SkyScript_cls']['test_data']
        classnames = BENCHMARK_DATASET_INFOMATION['SkyScript_cls']['classnames']

        # Prepare model name.
        model = 'ViT-L-14'  # MODEL_TYPE.
        # Prepare pretrained model direction.
        pretrained = PRETRAINED_VIT_MODEL

        # Construct args.
        arg_list = [
            '--root-data-dir=' + self.dir_root_dset,
            '--classification-mode=multiclass',
            '--csv-separator=,', 
            '--csv-img-key', 'filepath',  
            '--csv-class-key', 'label',   
            '--batch-size=128', 
            '--workers=8', 
            '--model=' + model, 
            '--pretrained=' + pretrained, 
            '--test-data=' + test_data, 
            '--classnames=' + classnames,
            '--test-data-name=' + 'SkyScript_cls',
            '--pred-csv-path=' + self.dirf_pred_csv,
        ]

        # This is because the ViT-L-14 model is initialized with the OpenAI model.
        if model == 'ViT-L-14':
            arg_list.append('--force-quick-gelu') 

        # Call test function to predict tiles' labels.
        # During the test process, the prediction results are saved under the specified path.
        test(arg_list)
            
    def capmap_disp(self, dprect=True, dppoint=True, dpindex=True, dplabel=True):
        """
        Read label list (Chinese/english options), 
        sequence index of tiles (To infer coordinate list), 
        prediction result file (To infer pred list),
        Read RS image, calculate its l-r-d-u latitude and longitude, 
        Plotting all tiles consists of label in different colors on the RS image.
        Save the caption scape image.
        You can chose witch info to show on the image by form paras:
            dprect=True  : display rect  on the caption scap image.
            dppoint=True : display point on the caption scap image.
            dpindex=True : display index on the caption scap image.
            dplabel=True : display label on the caption scap image.
        """
        print('\n=-        -=classCapMap/capmap_disp() is called.=-        -=')
        # Your input RS image path.
        dir_src  = self.dir_root_dset + '/' + self.name_dset_src  
        # Path ofyYour mini dataset crop from RS image.
        dir_dset = self.dir_root_dset + '/' + self.name_dataset  
        # Path of all results.
        dir_save_result = self.dir_root_dset + '/' + self.name_save_result  
        self._dir_check_and_make(dir_save_result)
            
        # Get the path of the relevant files under dataset_name
        dirf_txt_cls   = dir_dset + "/" + 'classnames.txt'
        dirf_txt_clscn = dir_dset + "/" + 'classnames_70cn.txt'
        dirf_csv_cls   = dir_dset + "/" + 'img_txt_pairs_val.csv'
        dirf_save_pic  = dir_save_result + '/' + self.name_result + '.png'
        
        # Read class label, Assembled into English Chinese dictionary.
        label_list   = self._classnames2list(dirf_txt_cls)
        label_listcn = self._classnames2list(dirf_txt_clscn)
        
        # Judge witch language to show.
        if self.label_language == 'chinese':
            dict_label_language = dict(zip(label_list, label_listcn))  
        elif self.label_language == 'english':
            dict_label_language = dict(zip(label_list, label_list))  
        else:
            pass  # Here, language options can be added on demand.
        
        # Read CSV files
        df_cls  = pd.read_csv(dirf_csv_cls)
        df_pred = pd.read_csv(self.dirf_pred_csv)
        
        # Get list of filepath infomation in df_cls.
        lst_filepath = df_cls.filepath.tolist()
        # e.g. XXX_cls/val/0_004-007_30.2989714,119.9424729.jpg.
        # Get list of coordination pair of each tile.
        lst_latlon_pair = [re.findall(r'\d+\.\d+', x) for x in lst_filepath]
        lst_latlon_pair = [[float(x) for x in y] for y in lst_latlon_pair]  
        print("===lst_latlon_pair:", lst_latlon_pair)
        
        # Get predictions and change to list form.
        lst_pred = df_pred['0'].tolist()

        # Read RS image
        dirf_rsimage = dir_src + '/' + self.name_xmap  #/home/lkj/dataset/arbm52_src
        background_img = mpimg.imread(dirf_rsimage)
        
        # Implement exact display for capmap and tiles.
        # Calculate the RSI's coordinate range, write to lst_angle_lrdu,
        # so that the displayed coordinates accurately match the RS image.
        print("===self.name_xmap:", self.name_xmap)
        if self.name_xmap:
            lst_angle_lrdu = self.lon_lat_of_map_corner(self.name_xmap)
            #e.g. 119.94125714250534 119.95328865749467 30.302218246846845 30.308524553153152.
        else:  
            # Backby strategy: obtained with first-end tile coordinates: 
            # lon_left, lon_right, lat_down, lat_up.
            lat_down, lon_left = lst_latlon_pair[0]
            lat_up, lon_right  = lst_latlon_pair[-1]
            lst_angle_lrdu = [lon_left, lon_right, lat_down, lat_up]
        
        # Create color dictionary of label_list.
        dct_color = self._get_color_dict_70sky(label_list)

        # Ploting. 
        # Order : Get color list, load the large picture, 
        #         draw tile square, center point and label number.
        figwidth = 12
        fig = plt.figure(figsize=(12, 6), dpi=300)
        ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
        ax1.imshow(background_img, extent=lst_angle_lrdu, alpha=0.9)
        
        # Plot the impact of classification to trajectories.  
        ax1.set_title(self.name_result, fontsize=20)
        for latlon_, pred_ in zip(lst_latlon_pair, lst_pred):
            
            if dprect:
                # Plot a semi-transparent square around the coordinate point.
                rectx = latlon_[1] - self.lon_angle_tile_span/2  #left of tile.
                recty = latlon_[0] - self.lon_angle_tile_span/2  #down of tile.
                # Add the rectangles to the coordinate axis.
                rect = patches.Rectangle(  
                    (rectx, recty),        
                    self.lon_angle_tile_span, 
                    self.lon_angle_tile_span,
                    linewidth=0.5, 
                    edgecolor='b',  
                    facecolor=dct_color[label_list[pred_]],  
                    alpha=0.15)
                ax1.add_patch(rect)   

            # Plot the center Point of tile.
            if dppoint:
                ax1.scatter(
                    latlon_[1], 
                    latlon_[0], 
                    marker=',', 
                    s=figwidth-2, 
                    facecolor=dct_color[label_list[pred_]])   
            
            # Plot predicted class label number.
            if dpindex:
                text = plt.text(
                    x=latlon_[1],
                    y=latlon_[0], 
                    s=str(pred_), 
                    fontdict=dict(fontsize=figwidth-9+self.tile_size, 
                                    color=dct_color[label_list[pred_]],
                                    family='monospace',))
            
            # Plot predicted class label title.
            if dplabel:
                dd = 0.01/30
                my_list = [1, 2, 3, 4, 5]
                font = FontProperties(fname=self.dir_root_dset+"/simsun.ttc", size=14)
                text = plt.text(
                    x=latlon_[1] - dd/3,
                    y=latlon_[0] - dd/10 * float(random.sample(my_list, 1)[0]), 
                    s=dict_label_language[label_list[pred_]], 
                    fontdict=dict(fontsize=figwidth-10+self.tile_size, 
                                color=dct_color[label_list[pred_]],
                                family='monospace',),
                    fontproperties=font)

        ax1.set_xlabel('longitude',fontsize=15,family='Arial')
        ax1.set_ylabel('latitude',fontsize=15,family='Arial') 
        plt.subplots_adjust()
        plt.tight_layout()
        
        # Saving the caption scape image.
        figdpi = 150
        plt.savefig(dirf_save_pic, 
                    dpi=figdpi * 2, 
                    bbox_inches='tight', 
                    pad_inches=0.0, 
                    edgecolor='black', 
                    facecolor='white')

    """Private functions"""
    def _open_src_image(self):
        """
        Open and output a remote sensing image.
        """
        try:
            # Dir of source RS image.
            self.dir_src_bigmaps  = self.dir_root_dset + '/' + self.name_dset_src  
            dirf_rsimage = ( self.dir_src_bigmaps + '/' + self.name_xmap)
            rsimage = Image.open(dirf_rsimage)
            return rsimage
        except IOError as e:
            print(f"Can not open the image: {e}")
            
    def _shorten_name_and_get_lonlat(self, name_rsimage):
        """
        Get the list of longitude and latitude coordinate pairs,
        and a short name with decimal points retain 3 bits.
        Input: '30.2613714,119.9232729.jpg'
        output: '30.305,119.947', [30.301,119.947]
        """
        # Get True/False, [30.3013714,119.9472729] below:
        latlon_rsi_cnt = self._latlon_rsi_cnt_from_imname(name_rsimage)
        # Generate a short name for name_rsimage to 
        # facilitate the simplicity of the results.
        print("=-=latlon_rsi_cnt:", latlon_rsi_cnt)
        lst_latlon_short = [str(f'{x:.3f}') for x in latlon_rsi_cnt]  
        # Keep 3 decimal places e.g., [30.301,119.947]
        shorten_name = ','.join(lst_latlon_short)  # '30.305,119.947'
        
        return shorten_name, latlon_rsi_cnt
    
    def _latlon_rsi_cnt_from_imname(self, name_rsimage):
        """
        Calculate the latitude and longitude of the RSI's center.
        """
        flag_name_coor = True  # coordinate info in image name.
        latlon_rsi_cnt = re.findall(r'\d+\.\d+', name_rsimage)
        
        # Two cases depend on whether the image name contains coordination.
        # If there is no coordinate info in image name:
        if len(latlon_rsi_cnt) == 0:
            # Then given a virtual coordinate pair:
            latlon_rsi_cnt = [30.0000, 30.0000]
            flag_name_coor = False  # NO coordinate info in image name.
        else:
            latlon_rsi_cnt = [float(x) for x in latlon_rsi_cnt]
        print("=-=a latlon_rsi_cnt:", latlon_rsi_cnt)
        return latlon_rsi_cnt
        
    def _get_color_dict_70sky(self, label_list):
        """
        Enter the list of category names and combine the cluster color table 
        with each category name to generate the color dictionary. 
        In this way, the color dictionary shows similar object colors, 
        which is easy to observe and improves the visualization effect.
        """
        # Base color, which will generate 70 different colors based on this color.
        base_color = [1, 0, 0]  

        # Generate color list with n colors.
        lst_color = self._generate_colors(base_color, len(label_list))
    
    
        # lst_color = colors_list_gen(len(label_list))
        # Category-color index table. LST_COLOR_IDX_70.
        lst_color_idx = [
            30, 31, 4, 63, 27, 0, 39, 25, 28, 17, 
            12, 40, 52, 44, 32, 60, 21, 65, 19, 3, 
            46, 69, 66, 62, 53, 22, 1, 61, 16, 59, 
            56, 45, 33, 67, 34, 38, 42, 36, 23, 51, 
            9, 18, 68, 49, 24, 7, 15, 41, 58, 10, 
            8, 2, 54, 64, 26, 43, 29, 5, 50, 11, 
            13, 48, 55, 47, 14, 35, 6, 57, 37, 20]
        # dct_color1 = dict(zip(label_list, lst_color))
        # Colors were adjusted for classification.
        dct_color = {}  
        for i, label_ in enumerate(label_list):
            dct_color[label_] = lst_color[lst_color_idx[i]]
        return dct_color
        
    def _generate_colors(self, base_color, n_colors):
        """
        Generate color list with n colors
        """
        hsv_colors = np.zeros((n_colors, 3))
        # Color phase from 0 to 1.
        hsv_colors[:, 0] = np.linspace(0, 1, n_colors) 
        # The saturation was 1. 
        hsv_colors[:, 1] = 1  
        # Brightness ranged from 0.2 to 1.
        hsv_colors[:, 2] = np.linspace(0.2, 1, n_colors)  
        
        # Convert the HSV color to an RGB color.
        rgb_colors = np.apply_along_axis(hsv_to_rgb, 1, hsv_colors)
        
        # Set base color.
        rgb_colors[0] = base_color
        return rgb_colors
        
    def _classnames2list(self, dirf_txt_cls):
        """
        Open txt file like 'classnames.txt', one category name per row,
        Convert it to the list output, e.g. ['clsA','clsB',...]
        """
        label_list = []
        with open(dirf_txt_cls, 'r') as f:
            for line in f:
                label_list.append(line.strip())
        return label_list

    def _write_list_to_csv_in_patha(self, lst, csvfilename, patha):
        """
        Write list to csv file and coexist to patha.
        """
        csvpath = patha + "/" + csvfilename
        
        # csv list head.
        lst_label_head = ['filepath', 'title', 'label']
        
        # write to file.
        with open(csvpath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(lst_label_head)
            for row in lst:
                writer.writerow(row)

    def _copy_srcfile_to_desfile(self, fs, fd):
        """
        Copy one file to another location,
        Enter is the path for both files.
        """
        # Full path of source file.
        source_file_path = fs  #'path/to/source/file.txt'
        # Full path to the target file.
        destination_file_path = fd  #'path/to/destination/file.txt'

        # Copy file.
        try:
            shutil.copy2(source_file_path, destination_file_path)
            print(f"-The file has been copied from {source_file_path} to {destination_file_path}.")
        except FileNotFoundError:
            print(f"-The source file {source_file_path} does not exist.")
        except Exception as e:
            print(f"-An error occurred: {e}")

    def _copy_files_from_pathA2B(self, lst_file, patha, pathb):
        """
        Copy the files under path A to path B, and 
        the list of files indicates moving those files.
        """
        for txtfile in lst_file:
            srctxtpath = patha + "/" + txtfile
            dsttxtpath = pathb + "/" + txtfile   
            print(f'-.-copy_files_from_pathA2B()/srctxtpath is {os.path.exists(srctxtpath)}')
            print(f'-.-copy_files_from_pathA2B()/dsttxtpath is {os.path.exists(dsttxtpath)}')
            # If there is a prepared source txt file "classnames.txt".
            if os.path.exists(srctxtpath) and not os.path.exists(dsttxtpath):
                # Copy classnames.txt to des dir.
                self._copy_srcfile_to_desfile(srctxtpath, dsttxtpath)
            else:  
                # If there is no source txt file, auto create the classnames.txt.
                # Auto chose classnames in english or in chinese.
                label_list = CLASSNAMES_70CN if 'cn' in txtfile else CLASSNAMES_70
                # Open a new TXT file
                with open(dsttxtpath, "w") as file:
                    # Write the list content to the file.
                    for item in label_list:
                        file.write(item + "\n")

    def _dir_check_and_make(self, dir_dset):
        """
        Check that the folder exists and create a new one.
        """
        if not os.path.exists(dir_dset):
            os.makedirs(dir_dset)
            print(f"Folder '{dir_dset}' was created.")
        else:
            print(f"Folder '{dir_dset}' already exists.")
            
    """calculations functions"""
    def lon_lat_of_map_corner(self,name_rsimage):
        """
        Calculate the latitude and longitude of the four corners 
        given the map for easy display.
        Input: 
            satellite map, 
            central longitude and latitude coordinates, 
            related information dictionary.
        The dictionary is designed in a fixed format for subsequent expansion.
        """
        # Central coordinates of the map.
        latlon_rsi_cnt = self._latlon_rsi_cnt_from_imname(name_rsimage)
        print('-=-latlon_rsi_cnt', latlon_rsi_cnt)
        
        # Known width and height of RS image, converted to the corresponding longitude and longitude.
        delta_lon_meter = self.RSI_WIDTH_METER  # unit meters. height value in meters.
        delta_lat_meter = self.RSI_HEIGHT_METER # unit meters. height value in meters.
        lon, lat = self.meter2angle(delta_lon_meter, delta_lat_meter, latlon_rsi_cnt[0])
        print('-=-lonlat_span', lon, lat)
        
        # Calculate the left and right sides of the RSI and the upper latitude.
        lon_left, lon_right = latlon_rsi_cnt[1] - lon/2, latlon_rsi_cnt[1] + lon/2
        lat_down, lat_up   = latlon_rsi_cnt[0] - lat/2, latlon_rsi_cnt[0] + lat/2
        print('-=-lrdu', [lon_left, lon_right, lat_down, lat_up])
        return [lon_left, lon_right, lat_down, lat_up]

    def meter2pixel(self,lon_meter, lat_meter):
        """
        Turn units of quantity from meters to pixels.
        """
        lat_mpp = self.RSI_LAT_METER_PER_PIXEL
        lon_mpp = self.RSI_LON_METER_PER_PIXEL
        lat_pixel = lat_meter // lat_mpp
        lon_pixel = lon_meter // lon_mpp    
        return lon_pixel, lat_pixel

    def pixel2meter(self,delta_lon_pixel, delta_lat_pixel):
        """
        Turn units of quantity from pixels to meters.
        """
        delta_lon_meter = delta_lon_pixel * self.RSI_LON_METER_PER_PIXEL
        delta_lat_meter = delta_lat_pixel * self.RSI_LAT_METER_PER_PIXEL
        return delta_lon_meter, delta_lat_meter

    def angle2meter(self,delta_lon_angle, delta_lat_angle, lat):
        """
        Turn units of angle from degree to meters. Need input latitude.
        """
        delta_lat_meter = delta_lat_angle * 111000  
        delta_lon_meter = delta_lon_angle * 111000 * math.cos(lat / 180 * math.pi)
        return delta_lon_meter, delta_lat_meter

    def meter2angle(self,delta_lon_meter, delta_lat_meter, lat):
        """
        Turn the input longitude and latitude in meters 
        to the corresponding longitude and latitude.
        """
        delta_lat_angle = delta_lat_meter / 111000
        delta_lon_angle = delta_lon_meter / (111000 * math.cos(lat / 180 * math.pi))
        return delta_lon_angle, delta_lat_angle


if __name__ == "__main__":
    
    # Before running this program, you have to 
    
    # Step1. Specify YOUR root direction parameter '--rootpath='.
    # The inputs and outputs can save to this direction.
    YOUR_ROOT_DIR = '/.../.../...'
    
    # Step2. Build a new document "rsimage" under your root dir
    # and copy a remote sensing image in it. The image should be
    # "*.jpg" or "*.png" or "*.bmp".
    
    # Step3. Specify YOUR direction of pretrained model. 
    PRETRAINED_VIT_MODEL = '/.../.../*.pt'

    # The other paragrames canbe defaulted in the first running.
    # You can adjust them according your own needs.
    # If you want to display chinese label characters, please 
    # download and copy "simsun.ttc" file to document "rsimage".
    
    
    arg_list_capmap = [
        '--rootpath=' + YOUR_ROOT_DIR,  # Root direction of your Remote Sensing Image.
        '--pretrained-model=' + PRETRAINED_VIT_MODEL,  # Root direction of your Remote Sensing Image.
        '--x=' + 'A',                         # The index of RS image, A...Z,a...z, Up to 52 RS images can be indexed. 
                                              # If you put two RS images in document 'rsimage', you got 2 options 'A' & 'B'.
        '--label-language=' + 'chinese',      # In which language are class names displayed on the CSI.
        '--s=8',                              # Scale factor of tile, Generally take value 3,4,5,6,7,8,9
        '--c=1',                              # Scale factor of crop, i.e., additional cuts at the edge of tile
        '--imagesrc=' + 'rsimage',            # The document contains source remote sensing image (RSI). 
        '--savepath=' + 'rsitile_pred',       # The document contains output caption scape image (CSI) and csv file.
        '--imset=' + 'rsi',                   # The prefix of automatically generated dataset from image tiles.
        ]
    
    mycm = CapMap(arg_list_capmap)
    mycm.capmap_crop()
    mycm.capmap_pred()
    mycm.capmap_disp()
    