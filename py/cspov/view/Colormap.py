#!/usr/bin/env python
# -*- coding, utf-8 -*-
"""
.py
~~~

PURPOSE


REFERENCES


REQUIRES


,author: R.K.Garcia <rayg@ssec.wisc.edu>
,copyright: 2014 by University of Wisconsin Regents, see AUTHORS for more details
,license: GPLv3, see LICENSE for more details
"""
__docformat__ = 'reStructuredText'
__author__ = 'cphillips'

from vispy.color.colormap import Colormap, BaseColormap, _mix_simple
from collections import OrderedDict
import numpy as np


class SquareRootColormap(BaseColormap):
    colors = [(0.0, 0.0, 0.0, 1.0),
              (1.0, 1.0, 1.0, 1.0)]

    glsl_map = """
    vec4 sqrt_cmap(float t) {
        return mix($color_0, $color_1, sqrt(t));
    }
    """

    def map(self, t):
        a, b = self.colors.rgba
        return _mix_simple(a, b, np.sqrt(t))


# Hand made colormap for the CIRA IR Default colormap
# The XML file from AWIPS doesn't separate things very well or in a way that can be easily automated
# we have to include every break point
cira_ir_colors = [(0.0, 0.0, 0.0, 1.0), # Black
          (0.980392156862745, 0.980392156862745, 0.980392156862745, 1.0), # White
          (0.972549019607843, 0.949019607843137, 0, 1.0), # Bright Yellow
          (0.372549019607843, 0.372549019607843, 0.0, 1.0), # Dark Yellow
          (0.564705882352941, 0.0, 0.0, 1.0), # Dark Red
          (0.929411764705882, 0.0, 0.0, 1.0), # Bright Red
          (0.949019607843137, 0.0, 0.972549019607843, 1.0), # Bright magenta
          (0.549019607843137, 0.0, 0.552941176470588, 1.0), # Dark magenta
          (0.0, 0.372549019607843, 0.564705882352941, 1.0), # Dark Cyan
          (0.0, 0.780392156862745, 0.815686274509804, 1.0), # Bright Cyan
          (0.0470588235294118, 0.972549019607843, 0.0, 1.0), # Bright Green
          (0.0, 0.435294117647059, 0.0, 1.0), # Dark Green
          (0.0235294117647059, 0.0, 0.0, 1.0), # Black
          (0.847058823529412, 0.847058823529412, 0.847058823529412, 1.0), # white-ish
          (0.00392156862745098, 0.0, 0.972549019607843, 1.0), # Bright Blue
          (0.0, 0.0, 0.4, 1.0), # Dark Blue
          (0.972549019607843, 0.0, 0.466666666666667, 1.0), # Bright Red
          (0.0980392156862745, 0.0, 0.0392156862745098, 1.0), # Dark Red
          (1.0, 1.0, 1.0, 1.0),
          (1.0, 1.0, 1.0, 1.0),
          ]

cira_ir_segment_indexes = [
    55,
    -30,
    -40,
    -50,
    -60,
    -70,
    -80,
    -90,
    -100,
    -108,
    -109,
]
cira_ir_controls = [(y - 55)/(-109 - 55) for x in cira_ir_segment_indexes for y in [x, x]][1:-1]

cira_ir_default = Colormap(cira_ir_colors, controls=cira_ir_controls)



# Modified cloud_amount_default with 1.0 bounds value
_cloud_amount_default_control_points = (0.0, 0.03137254901960784, 0.2196078431372549, 0.23137254901960785, 0.2627450980392157, 0.29411764705882354, 0.2980392156862745, 0.996078431372549, 1.0)
_cloud_amount_default_colors = ('#000000', '#181818', '#bababa', '#c4c4c4', '#e0e0e0', '#fbfbfb', '#ffa67d', '#ffff00', '#ffff00')
cloud_amount_default = Colormap(colors=_cloud_amount_default_colors, controls=_cloud_amount_default_control_points)
_cloud_top_height_control_points = (0.0, 0.2901960784313726, 0.4549019607843137, 0.611764705882353, 0.7843137254901961, 0.8, 0.8156862745098039, 0.8235294117647058, 0.8627450980392157, 0.9019607843137255, 1.0)
_cloud_top_height_colors = ('#000000', '#020202', '#ac7070', '#a18331', '#41a166', '#211d8a', '#211d89', '#05f7fb', '#fbf702', '#fb0000', '#ffffff')
cloud_top_height = Colormap(colors=_cloud_top_height_colors, controls=_cloud_top_height_control_points)
# Modified low_cloud_base with 1.0 bounds value
_low_cloud_base_control_points = (0.0, 0.11764705882352941, 0.12549019607843137, 0.1450980392156863, 0.21176470588235294, 0.30196078431372547, 0.3058823529411765, 0.4980392156862745, 0.5450980392156862, 0.5490196078431373, 0.7843137254901961, 1.0)
_low_cloud_base_colors = ('#00aac7', '#008cab', '#ffffff', '#0087a6', '#007597', '#005a82', '#005981', '#7f7f7f', '#8a8a8a', '#8b8b8b', '#ff0000', '#ff0000')
low_cloud_base = Colormap(colors=_low_cloud_base_colors, controls=_low_cloud_base_control_points)
_rain_rate_control_points = (0.0, 0.5215686274509804, 0.6, 0.6470588235294118, 0.6980392156862745, 0.7490196078431373, 0.8, 0.8392156862745098, 0.8431372549019608, 0.8941176470588236, 1.0)
_rain_rate_colors = ('#000000', '#c7c7c7', '#00ffff', '#0000ff', '#00ff00', '#ffff00', '#ff9500', '#e20000', '#f00000', '#ff00ff', '#ffffff')
rain_rate = Colormap(colors=_rain_rate_colors, controls=_rain_rate_control_points)

_color11new_control_points = (0.0, 0.0019540791402051783, 0.1265266243282853, 0.2510991695163654, 0.37567171470444555, 0.43820224719101125, 0.5007327796775769, 0.625305324865657, 0.6878358573522227, 0.7503663898387885, 1.0)
_color11new_colors = ('#7e0000', '#190000', '#fe0000', '#fffe00', '#00ff00', '#00807e', '#00fefe', '#0000ff', '#7e0080', '#fe00fe', '#000000')
color11new = Colormap(colors=_color11new_colors, controls=_color11new_control_points)
# _cira_ir_default_control_points = (0.0, 0.6627450980392157, 0.7215686274509804, 0.7254901960784313, 0.7607843137254902, 0.8, 0.803921568627451, 0.8470588235294118, 0.8901960784313725, 0.9215686274509803, 0.9568627450980393, 0.9921568627450981, 1.0)
# _cira_ir_default_colors = ('#000000', '#f9f9f9', '#686600', '#5e5a00', '#e20000', '#97009b', '#8b008c', '#00c6d0', '#006f00', '#d8d8d8', '#000077', '#360017', '#ffffff')
# cira_ir_default = Colormap(colors=_cira_ir_default_colors, controls=_cira_ir_default_control_points)
_fog_control_points = (0.0, 0.027450980392156862, 0.17647058823529413, 0.24313725490196078, 0.3254901960784314, 0.3764705882352941, 1.0)
_fog_colors = ('#000000', '#0c0c0c', '#4e4e4e', '#6d6d6d', '#919191', '#a9a9a9', '#ffffff')
fog = Colormap(colors=_fog_colors, controls=_fog_control_points)
_ir_wv_control_points = (0.0, 0.6431372549019608, 0.7607843137254902, 0.8, 0.803921568627451, 0.8470588235294118, 0.8901960784313725, 0.9215686274509803, 0.9568627450980393, 0.9921568627450981, 1.0)
_ir_wv_colors = ('#000000', '#f5f5f5', '#df0000', '#94009a', '#8a008a', '#00c4ce', '#006d00', '#d5d5d5', '#000076', '#350016', '#ffffff')
ir_wv = Colormap(colors=_ir_wv_colors, controls=_ir_wv_control_points)
_lifted_index__new_cimss_table_control_points = (0.0, 0.4823529411764706, 0.5607843137254902, 0.5647058823529412, 0.6392156862745098, 0.6431372549019608, 0.7176470588235294, 0.7215686274509804, 0.7843137254901961, 0.9921568627450981, 1.0)
_lifted_index__new_cimss_table_colors = ('#000000', '#412a0e', '#7876cf', '#7b79d4', '#767502', '#706f02', '#ea0000', '#f10000', '#f575c7', '#c6c6c5', '#ffffff')
lifted_index__new_cimss_table = Colormap(colors=_lifted_index__new_cimss_table_colors, controls=_lifted_index__new_cimss_table_control_points)
_lifted_index_default_control_points = (0.0, 0.20392156862745098, 0.2901960784313726, 0.29411764705882354, 0.34901960784313724, 0.35294117647058826, 0.40784313725490196, 0.4627450980392157, 0.4666666666666667, 0.5215686274509804, 0.6392156862745098, 0.6431372549019608, 0.6941176470588235, 0.7803921568627451, 1.0)
_lifted_index_default_colors = ('#000000', '#7f1818', '#0abf06', '#07c405', '#7382c3', '#7b8cc6', '#ca9444', '#bac005', '#bec403', '#ca84c0', '#c37801', '#c67b01', '#b200b6', '#c27ec4', '#ffffff')
lifted_index_default = Colormap(colors=_lifted_index_default_colors, controls=_lifted_index_default_control_points)
# Modified blended_total_precip_water with 1.0 bounds value
_blended_total_precip_water_control_points = (0.0, 0.011764705882352941, 0.7372549019607844, 0.7764705882352941, 0.7843137254901961, 0.8313725490196079, 0.8352941176470589, 0.8588235294117647, 0.8823529411764706, 0.9294117647058824, 0.9333333333333333, 0.9803921568627451, 0.9882352941176471, 1.0)
_blended_total_precip_water_colors = ('#000000', '#db9270', '#0000c4', '#00e7e7', '#00ffff', '#00f300', '#00ff00', '#b9b900', '#ffff00', '#f30000', '#ff0000', '#f300f3', '#ffffff', '#ffffff')
blended_total_precip_water = Colormap(colors=_blended_total_precip_water_colors, controls=_blended_total_precip_water_control_points)
# Modified percent_of_normal_tpw with 1.0 bounds value
_percent_of_normal_tpw_control_points = (0.0, 0.12156862745098039, 0.13333333333333333, 0.17254901960784313, 0.29411764705882354, 0.5843137254901961, 0.6431372549019608, 0.7019607843137254, 0.7098039215686275, 0.7450980392156863, 0.7764705882352941, 0.7803921568627451, 0.984313725490196, 1.0)
_percent_of_normal_tpw_colors = ('#000000', '#875a45', '#94634c', '#c08062', '#ecc9b7', '#05ffff', '#00dbdb', '#00b4b4', '#00b0b0', '#009999', '#008484', '#008181', '#ffffff', '#ffffff')
percent_of_normal_tpw = Colormap(colors=_percent_of_normal_tpw_colors, controls=_percent_of_normal_tpw_control_points)
_precip_water__new_cimss_table_control_points = (0.0, 0.10980392156862745, 0.23137254901960785, 0.34901960784313724, 0.4627450980392157, 0.5764705882352941, 0.5803921568627451, 0.6941176470588235, 0.788235294117647, 0.9921568627450981, 1.0)
_precip_water__new_cimss_table_colors = ('#000000', '#4c3516', '#9390f2', '#0a675e', '#9fc33c', '#f3775c', '#f77a5e', '#700071', '#ffc2fc', '#c6c6c6', '#ffffff')
precip_water__new_cimss_table = Colormap(colors=_precip_water__new_cimss_table_colors, controls=_precip_water__new_cimss_table_control_points)
_precip_water__polar_control_points = (0.0, 0.6862745098039216, 0.7333333333333333, 0.7372549019607844, 0.796078431372549, 0.8156862745098039, 0.8509803921568627, 0.8745098039215686, 0.9294117647058824, 1.0)
_precip_water__polar_colors = ('#000000', '#ffffff', '#f1f100', '#ffff00', '#00bbef', '#00ffff', '#aa0000', '#ff0000', '#c7ffc7', '#ffffff')
precip_water__polar = Colormap(colors=_precip_water__polar_colors, controls=_precip_water__polar_control_points)
_precip_water_default_control_points = (0.0, 0.09411764705882353, 0.21176470588235294, 0.25098039215686274, 0.2901960784313726, 0.32941176470588235, 0.3686274509803922, 0.40784313725490196, 0.4823529411764706, 0.48627450980392156, 0.49411764705882355, 0.4980392156862745, 0.7803921568627451, 1.0)
_precip_water_default_colors = ('#000000', '#850200', '#ca2b00', '#ca9117', '#6bac1f', '#22ac4a', '#279fa1', '#821ba1', '#2629a6', '#1e1fa1', '#ca4205', '#ca8203', '#070355', '#ffffff')
precip_water_default = Colormap(colors=_precip_water_default_colors, controls=_precip_water_default_control_points)
_skin_temp__new_cimss_table_control_points = (0.0, 0.17254901960784313, 0.17647058823529413, 0.25098039215686274, 0.2549019607843137, 0.3333333333333333, 0.40784313725490196, 0.48627450980392156, 0.49019607843137253, 0.6431372549019608, 0.7764705882352941, 0.7803921568627451, 1.0)
_skin_temp__new_cimss_table_colors = ('#000000', '#cc5111', '#cc5211', '#f6e207', '#faeb07', '#1aaf03', '#03edd9', '#0523ff', '#0517ff', '#fcedff', '#1afcff', '#13fcff', '#ffffff')
skin_temp__new_cimss_table = Colormap(colors=_skin_temp__new_cimss_table_colors, controls=_skin_temp__new_cimss_table_control_points)
_skin_temp_default_control_points = (0.0, 0.2901960784313726, 0.2980392156862745, 0.3764705882352941, 0.5333333333333333, 0.6901960784313725, 0.8, 0.8156862745098039, 0.8235294117647058, 0.8627450980392157, 0.9019607843137255, 1.0)
_skin_temp_default_colors = ('#000000', '#020202', '#9d3c5e', '#9d4c00', '#9d9900', '#1a8c63', '#211d8b', '#211d8a', '#05f8fc', '#fcf802', '#fc0000', '#ffffff')
skin_temp_default = Colormap(colors=_skin_temp_default_colors, controls=_skin_temp_default_control_points)
_ca_low_light_vis_control_points = (0.0, 0.3333333333333333, 0.6901960784313725, 0.7333333333333333, 0.7764705882352941, 1.0)
_ca_low_light_vis_colors = ('#000000', '#818181', '#ffffff', '#9b9b9b', '#464646', '#777777')
ca_low_light_vis = Colormap(colors=_ca_low_light_vis_colors, controls=_ca_low_light_vis_control_points)
_linear_control_points = (0.0, 0.2980392156862745, 0.4980392156862745, 0.6980392156862745, 0.7058823529411765, 0.8862745098039215, 1.0)
_linear_colors = ('#000000', '#4c4c4c', '#7f7f7f', '#b1b1b1', '#b4b4b4', '#e2e2e2', '#ffffff')
linear = Colormap(colors=_linear_colors, controls=_linear_control_points)
# Modified za_vis_default with 1.0 bounds value
_za_vis_default_control_points = (0.0, 0.23921568627450981, 0.2627450980392157, 0.34901960784313724, 0.5176470588235295, 0.8627450980392157, 1.0)
_za_vis_default_colors = ('#000000', '#0b0b0b', '#181818', '#4c4c4c', '#848484', '#ffffff', '#ffffff')
za_vis_default = Colormap(colors=_za_vis_default_colors, controls=_za_vis_default_control_points)
_gray_scale_water_vapor_control_points = (0.0, 0.6274509803921569, 0.7176470588235294, 0.7803921568627451, 0.9803921568627451, 0.9921568627450981, 1.0)
_gray_scale_water_vapor_colors = ('#000000', '#2e2e2e', '#606060', '#838383', '#f0f0f0', '#f6f6f6', '#ffffff')
gray_scale_water_vapor = Colormap(colors=_gray_scale_water_vapor_colors, controls=_gray_scale_water_vapor_control_points)
_nssl_vas_wv_alternate_control_points = (0.0, 0.42745098039215684, 0.47058823529411764, 0.5490196078431373, 0.6274509803921569, 0.6588235294117647, 0.6745098039215687, 0.6901960784313725, 0.7058823529411765, 0.7843137254901961, 0.9098039215686274, 1.0)
_nssl_vas_wv_alternate_colors = ('#000000', '#9e0000', '#7f0000', '#9e3f00', '#9e7f00', '#529e00', '#2bad00', '#00be45', '#009e6a', '#006a9e', '#a6a6a6', '#ffffff')
nssl_vas_wv_alternate = Colormap(colors=_nssl_vas_wv_alternate_colors, controls=_nssl_vas_wv_alternate_control_points)
_ramsdis_wv_control_points = (0.0, 0.7450980392156863, 0.8, 0.8392156862745098, 0.8745098039215686, 0.9019607843137255, 0.9098039215686274, 0.9137254901960784, 0.9568627450980393, 1.0)
_ramsdis_wv_colors = ('#000000', '#e5e5e5', '#0200ee', '#02ee02', '#df0000', '#ac4801', '#ffffff', '#626148', '#606060', '#ffffff')
ramsdis_wv = Colormap(colors=_ramsdis_wv_colors, controls=_ramsdis_wv_control_points)
_slc_wv_control_points = (0.0, 0.5529411764705883, 0.6431372549019608, 0.7686274509803922, 0.8196078431372549, 0.8235294117647058, 0.8274509803921568, 0.8627450980392157, 0.8862745098039215, 0.8901960784313725, 0.9137254901960784, 0.9254901960784314, 1.0)
_slc_wv_colors = ('#000000', '#995c35', '#3a3835', '#759285', '#166f4c', '#076a44', '#036842', '#008484', '#00b0b0', '#00b6b6', '#009300', '#006c00', '#ffffff')
slc_wv = Colormap(colors=_slc_wv_colors, controls=_slc_wv_control_points)

VIS_COLORMAPS = OrderedDict([
    ('CA (Low Light Vis)', ca_low_light_vis),
    ('Linear', linear),
    ('ZA', za_vis_default),
    ('Square Root (Vis Default)', SquareRootColormap()),
])

IR_COLORMAPS = OrderedDict([
    ('Rainbow (IR Default)', color11new),
    ('CIRA IR', cira_ir_default),
    ('Fog', fog),
    ('IR WV', ir_wv),
])

LIFTED_INDEX_COLORMAPS = OrderedDict([
    ('Lifted Index (New CIMSS)', lifted_index__new_cimss_table),
    ('Lifted Index', lifted_index_default),
])

PRECIP_COLORMAPS = OrderedDict([
    ('Blended TPW', blended_total_precip_water),
    ('Percent of Normal TPW', percent_of_normal_tpw),
    ('Precipitable Water (New CIMSS)', precip_water__new_cimss_table),
    ('Precipitable Water (Polar)', precip_water__polar),
    ('Precipitable Water', precip_water_default),
])

SKIN_TEMP_COLORMAPS = OrderedDict([
    ('Skin Temp (New CIMSS)', skin_temp__new_cimss_table),
    ('Skin Temp', skin_temp_default),
])

WV_COLORMAPS = OrderedDict([
    ('Gray Scale Water Vapor', gray_scale_water_vapor),
    ('NSSL VAS (WV Alternate)', nssl_vas_wv_alternate),
    ('RAMSDIS WV', ramsdis_wv),
    ('SLC WV', slc_wv),
])

OTHER_COLORMAPS = OrderedDict([
    ('Rain Rate', rain_rate),
    ('Low Cloud Base', low_cloud_base),
    ('Cloud Amount', cloud_amount_default),
    ('Cloud Top Height', cloud_top_height),
])

ALL_COLORMAPS = {}
ALL_COLORMAPS.update(VIS_COLORMAPS)
ALL_COLORMAPS.update(IR_COLORMAPS)
ALL_COLORMAPS.update(LIFTED_INDEX_COLORMAPS)
ALL_COLORMAPS.update(PRECIP_COLORMAPS)
ALL_COLORMAPS.update(SKIN_TEMP_COLORMAPS)
ALL_COLORMAPS.update(WV_COLORMAPS)
ALL_COLORMAPS.update(OTHER_COLORMAPS)

CATEGORIZED_COLORMAPS = OrderedDict([
    ("Visible", VIS_COLORMAPS),
    ("IR", IR_COLORMAPS),
    ("Lifted Index", LIFTED_INDEX_COLORMAPS),
    ("Precipitable Water", PRECIP_COLORMAPS),
    ("Skin Temperature", SKIN_TEMP_COLORMAPS),
    ("Water Vapor", WV_COLORMAPS),
    ("Other", OTHER_COLORMAPS),
])

DEFAULT_VIS = "Square Root (Vis Default)"
DEFAULT_IR = "Rainbow (IR Default)"