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

from vispy.color import Colormap
from collections import OrderedDict

_cloud_amount_default_control_points = (
    0.0,
    0.12156862745098039,
    0.1803921568627451,
    0.23921568627450981,
    0.996078431372549,
    1.0  # Nedded to be added
)

_cloud_amount_default_colors = ('#000000',
                                '#666666',
                                '#999999',
                                '#cccccc',
                                '#ffff00',
                                '#ffff00'  # Needed to be added
                                )

_cloud_top_height_control_points = (0.0,
                                    0.00784313725490196,
                                    0.10980392156862745,
                                    0.20392156862745098,
                                    1.0)
_cloud_top_height_colors = ('#000000',
                            '#999999',
                            '#666666',
                            '#333333',
                            '#ffffff')

_low_cloud_base_control_points = (0.0,  # Needed to be added since cmap must include all values 0 to 1
                                  0.12549019607843137,
                                  0.396078431372549,
                                  0.7843137254901961,
                                  1.0  # Needed to be added too
                                  )
_low_cloud_base_colors = ('#ffffff',
                          '#ffffff',
                          '#666666',
                          '#ff0000',
                          '#ff0000')

_rain_rate_control_points = (
    0.0,
    0.6,
    0.6470588235294118,
    0.6980392156862745,
    0.7490196078431373,
    0.8941176470588236,
    0.9725490196078431,
    1.0)
_rain_rate_colors = ('#000000',
                     '#00ffff',
                     '#0000ff',
                     '#00ff00',
                     '#ffff00',
                     '#ff00ff',
                     '#ffffff',
                     '#ffffff')

cloud_amount_default = Colormap(colors=_cloud_amount_default_colors, controls=_cloud_amount_default_control_points)
cloud_top_height = Colormap(colors=_cloud_top_height_colors, controls=_cloud_top_height_control_points)
low_cloud_base = Colormap(colors=_low_cloud_base_colors, controls=_low_cloud_base_control_points)
rain_rate = Colormap(colors=_rain_rate_colors, controls=_rain_rate_control_points)



# IR
_cira_ir_default_control_points = (0.0, 0.00392156862745098, 0.1411764705882353, 0.5411764705882353, 0.7529411764705882, 0.9607843137254902, 1.0)
_cira_ir_default_colors = ('#000000', '#000000', '#333333', '#cccccc', '#cc0000', '#000066', '#ffffff')
_fog_control_points = (0.0, 0.22745098039215686, 0.3411764705882353, 0.47058823529411764, 1.0)
_fog_colors = ('#000000', '#666666', '#999999', '#cccccc', '#ffffff')
_ir_wv_control_points = (0.0, 0.00392156862745098, 0.4470588235294118, 0.4980392156862745, 1.0)
_ir_wv_colors = ('#000000', '#000000', '#333333', '#666666', '#ffffff')

# Lifted Index
_lifted_index__new_cimss_table_control_points = (0.0, 1.0)
_lifted_index__new_cimss_table_colors = ('#000000', '#ffffff')
_lifted_index_default_control_points = (0.0, 1.0)
_lifted_index_default_colors = ('#000000', '#ffffff')

# Precip Water
# Modified blended_total_precip_water with 1.0 bounds value
_blended_total_precip_water_control_points = (0.0, 0.01568627450980392, 0.7843137254901961, 0.8352941176470589, 0.8823529411764706, 0.9333333333333333, 0.984313725490196, 0.9882352941176471, 1.0)
_blended_total_precip_water_colors = ('#000000', '#000000', '#00ffff', '#00ff00', '#ffff00', '#ff0000', '#ff00ff', '#ffffff', '#ffffff')
# Modified percent_of_normal_tpw with 1.0 bounds value
_percent_of_normal_tpw_control_points = (0.0, 0.39215686274509803, 0.43137254901960786, 0.47058823529411764, 0.5098039215686274, 0.5490196078431373, 0.5882352941176471, 0.6666666666666666, 0.7450980392156863, 0.984313725490196, 1.0)
_percent_of_normal_tpw_colors = ('#000000', '#ffffff', '#ccffff', '#99ffff', '#66ffff', '#33ffff', '#00ffff', '#00cccc', '#009999', '#ffffff', '#ffffff')
_precip_water__new_cimss_table_control_points = (0.0, 0.8235294117647058, 1.0)
_precip_water__new_cimss_table_colors = ('#000000', '#999999', '#ffffff')
_precip_water__polar_control_points = (0.0, 0.13725490196078433, 0.27450980392156865, 0.4117647058823529, 0.5490196078431373, 0.6862745098039216, 0.7372549019607844, 0.8156862745098039, 0.8745098039215686, 0.9882352941176471, 1.0)
_precip_water__polar_colors = ('#000000', '#333333', '#666666', '#999999', '#cccccc', '#ffffff', '#ffff00', '#00ffff', '#ff0000', '#ffffff', '#ffffff')
_precip_water_default_control_points = (0.0, 0.8235294117647058, 1.0)
_precip_water_default_colors = ('#000000', '#999999', '#ffffff')

# Skin Temp
_skin_temp__new_cimss_table_control_points = (0.0, 0.6509803921568628, 0.8980392156862745, 0.9490196078431372, 1.0)
_skin_temp__new_cimss_table_colors = ('#000000', '#ffffff', '#999999', '#cccccc', '#ffffff')
_skin_temp_default_control_points = (0.0, 0.011764705882352941, 0.10980392156862745, 0.20392156862745098, 1.0)
_skin_temp_default_colors = ('#000000', '#999999', '#666666', '#333333', '#ffffff')

# VIS
# Modified ca_low_light_vis with 1.0 bounds value
_ca_low_light_vis_control_points = (0.0, 0.00392156862745098, 0.20784313725490197, 0.2901960784313726, 0.37254901960784315, 0.4823529411764706, 0.6901960784313725, 0.8, 0.8470588235294118, 1.0)
_ca_low_light_vis_colors = ('#000000', '#000000', '#333333', '#666666', '#999999', '#cccccc', '#ffffff', '#000000', '#ffffff', '#ffffff')
_linear_control_points = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
_linear_colors = ('#000000', '#333333', '#666666', '#999999', '#cccccc', '#ffffff')
# Modified za_vis_default with 1.0 bounds value
_za_vis_default_control_points = (0.0, 0.00392156862745098, 0.3058823529411765, 0.39215686274509803, 0.592156862745098, 0.7843137254901961, 0.8627450980392157, 1.0)
_za_vis_default_colors = ('#000000', '#000000', '#333333', '#666666', '#999999', '#cccccc', '#ffffff', '#ffffff')

# WV
_gray_scale_water_vapor_control_points = (0.0, 0.00392156862745098, 0.7254901960784313, 0.8196078431372549, 1.0)
_gray_scale_water_vapor_colors = ('#000000', '#000000', '#666666', '#999999', '#ffffff')
_nssl_vas_wv_alternate_control_points = (0.0, 0.792156862745098, 1.0)
_nssl_vas_wv_alternate_colors = ('#000000', '#000000', '#ffffff')
_ramsdis_wv_control_points = (0.0, 0.7411764705882353, 0.9098039215686274, 0.9215686274509803, 1.0)
_ramsdis_wv_colors = ('#000000', '#cccccc', '#ffffff', '#ffffff', '#ffffff')
_slc_wv_control_points = (0.0, 0.7294117647058823, 0.8705882352941177, 1.0)
_slc_wv_colors = ('#000000', '#999999', '#009999', '#ffffff')

cira_ir_default = Colormap(colors=_cira_ir_default_colors, controls=_cira_ir_default_control_points)
fog = Colormap(colors=_fog_colors, controls=_fog_control_points)
ir_wv = Colormap(colors=_ir_wv_colors, controls=_ir_wv_control_points)
lifted_index__new_cimss_table = Colormap(colors=_lifted_index__new_cimss_table_colors, controls=_lifted_index__new_cimss_table_control_points)
lifted_index_default = Colormap(colors=_lifted_index_default_colors, controls=_lifted_index_default_control_points)
blended_total_precip_water = Colormap(colors=_blended_total_precip_water_colors, controls=_blended_total_precip_water_control_points)
percent_of_normal_tpw = Colormap(colors=_percent_of_normal_tpw_colors, controls=_percent_of_normal_tpw_control_points)
precip_water__new_cimss_table = Colormap(colors=_precip_water__new_cimss_table_colors, controls=_precip_water__new_cimss_table_control_points)
precip_water__polar = Colormap(colors=_precip_water__polar_colors, controls=_precip_water__polar_control_points)
precip_water_default = Colormap(colors=_precip_water_default_colors, controls=_precip_water_default_control_points)
skin_temp__new_cimss_table = Colormap(colors=_skin_temp__new_cimss_table_colors, controls=_skin_temp__new_cimss_table_control_points)
skin_temp_default = Colormap(colors=_skin_temp_default_colors, controls=_skin_temp_default_control_points)
ca_low_light_vis = Colormap(colors=_ca_low_light_vis_colors, controls=_ca_low_light_vis_control_points)
linear = Colormap(colors=_linear_colors, controls=_linear_control_points)
za_vis_default = Colormap(colors=_za_vis_default_colors, controls=_za_vis_default_control_points)
gray_scale_water_vapor = Colormap(colors=_gray_scale_water_vapor_colors, controls=_gray_scale_water_vapor_control_points)
nssl_vas_wv_alternate = Colormap(colors=_nssl_vas_wv_alternate_colors, controls=_nssl_vas_wv_alternate_control_points)
ramsdis_wv = Colormap(colors=_ramsdis_wv_colors, controls=_ramsdis_wv_control_points)
slc_wv = Colormap(colors=_slc_wv_colors, controls=_slc_wv_control_points)


_sqroot12_control_points = (0.0, 0.0009765625, 0.00390625, 0.0087890625, 0.015625, 0.0244140625, 0.03515625, 0.0478515625, 0.0625, 0.0791015625, 0.09765625, 0.1181640625, 0.140625, 0.1650390625, 0.19140625, 0.2197265625, 0.25, 0.2822265625, 0.31640625, 0.3525390625, 0.390625, 0.4306640625, 0.47265625, 0.5166015625, 0.5625, 0.6103515625, 0.66015625, 0.7119140625, 0.765625, 0.8212890625, 0.87890625, 0.9384765625, 1.0)
_sqroot12_colors = ('#000000', '#070707', '#0f0f0f', '#171717', '#1f1f1f', '#272727', '#2f2f2f', '#373737', '#3f3f3f', '#474747', '#4f4f4f', '#575757', '#5f5f5f', '#676767', '#6f6f6f', '#777777', '#7f7f7f', '#878787', '#8f8f8f', '#979797', '#9f9f9f', '#a7a7a7', '#afafaf', '#b7b7b7', '#bfbfbf', '#c7c7c7', '#cfcfcf', '#d7d7d7', '#dfdfdf', '#e7e7e7', '#efefef', '#f7f7f7', '#ffffff')
sqroot12 = Colormap(colors=_sqroot12_colors, controls=_sqroot12_control_points)

VIS_COLORMAPS = OrderedDict([
    ('CA (Low Light Vis)', ca_low_light_vis),
    ('Linear', linear),
    ('ZA (Vis Default)', za_vis_default),
    ('Square Root', sqroot12),
])

IR_COLORMAPS = OrderedDict([
    ('CIRCA IR (IR Default)', cira_ir_default),
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

DEFAULT_VIS = "ZA (Vis Default)"
DEFAULT_IR = "CIRCA IR (IR Default)"