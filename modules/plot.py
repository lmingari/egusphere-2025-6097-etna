import cartopy.feature as cfeature

def set_map(ax):
    BORDERS = cfeature.NaturalEarthFeature(
        scale     = '10m',
        category  = 'cultural',
        name      = 'admin_0_countries',
        edgecolor = 'gray',
        facecolor = 'none'
        )
    LAND = cfeature.NaturalEarthFeature(
        'physical', 'land', '10m',
        edgecolor = 'none',
        facecolor = 'lightgrey',
        alpha     = 0.8
        )
    ax.add_feature(LAND, zorder=0)
    ax.add_feature(BORDERS, linewidth=0.4)
    
def get_cbar_size(ax):
    bbox0 = ax.get_position()
    top_row_bottom = bbox0.y0 
    top_row_top = bbox0.y1

    cbar_width = 0.012
    cbar_height = 0.4

    cbar_center_y = (top_row_top + top_row_bottom) / 2
    cbar_bottom_y = cbar_center_y - (cbar_height / 2)

    return cbar_bottom_y, cbar_width, cbar_height