#!/usr/bin/env python
# coding: utf-8

# # Meshing RGI polygons
# 
# Here we'll show how to generate meshes out of polygons from the [Randolph Glacier Inventory](https://www.glims.org/RGI/) or RGI.
# The RGI is a collection of high-resolution digitized outlines of every mountain glacier on earth.
# There's a bit of preliminary data munging necessary to make things go well, which we'll demonstrate below.
# The entire inventory is a gigantic file, so in order to make the search and processing faster we'll fetch only the regional segment for Alaska.

# In[ ]:


import icepack
rgi_filename = icepack.datasets.fetch_randolph_glacier_inventory("alaska")


# This one shapefile contains many glacier outlines.
# Rather than go through every entry manually, we'll use geopandas to search for the glacier we want by name.

# In[ ]:


import geopandas
dataframe = geopandas.read_file(rgi_filename)


# We won't use this here, but it's good to see what's contained in each record.
# The inventory includes not just the glacier outlines but their area, slope, aspect, and elevation.
# So if you want to find (for example) the steepest glaciers in a particular region you can do that with a simple query.

# In[ ]:


dataframe.keys()


# Here we'll look at Gulkana Glacier, which is in the Alaska Range.

# In[ ]:


entry = dataframe[dataframe["glac_name"] == "Gulkana Glacier"]


# By default, the geometries in the RGI are stored in lat-lon coordinates, which isn't that useful to us.

# In[ ]:


outline_lat_lon = entry.geometry
outline_lat_lon.crs


# The geopandas API includes functions that will estimate which Universal Transverse Mercator zone the polygon will be in.
# In this case, Gulkana is in UTM zone 6.

# In[ ]:


utm_crs = outline_lat_lon.estimate_utm_crs()
utm_crs


# We can then convert the lat/lon geometry to the new coordinate system.
# Note that we don't necessarily need to use UTM zone 6.
# For example, you might be working with a gridded data set of, say, ice thickness or velocity that happens to be in a different UTM zone.
# In that case you should use whichever zone the rest of your data uses.

# In[ ]:


outline_utm = outline_lat_lon.to_crs(utm_crs)


# Next, all the meshing routines in icepack expect a GeoJSON file.
# The code below will convert the geopandas geometry into JSON, which has the undesirable effect of adding a superfluous Z coordinate.
# We can then use the `map_tuples` function from the GeoJSON library to strip this out.

# In[ ]:


import geojson
outline_json = geojson.loads(outline_utm.to_json())
outline = geojson.utils.map_tuples(lambda x: x[:2], outline_json)


# The icepack meshing module includes routines that will transform a GeoJSON data structure into the input format for a mesh generator like gmsh.

# In[ ]:


geometry = icepack.meshing.collection_to_geo(outline)
geo_filename = "gulkana.geo"
with open(geo_filename, "w") as geo_file:
    geo_file.write(geometry.get_code())


# Next we'll call the mesh generator at the command line.

# In[ ]:


get_ipython().system('gmsh -2 -v 0 gulkana.geo gulkana.msh')


# Finally, we'll load the result with Firedrake and visualize it.

# In[ ]:


import firedrake
mesh = firedrake.Mesh("gulkana.msh")


# In[ ]:


import matplotlib.pyplot as plt
fig, axes = plt.subplots()
axes.set_aspect("equal")
firedrake.triplot(mesh, axes=axes, interior_kw={"linewidth": 0.25})
axes.legend();


# The legend shows the numeric IDs that are used to define each segment of the boundary.
# You'll need these in the event that you need to define different boundary conditions on different segments, although for mountain glaciers it's usually enough to fix the velocity to zero at the boundaries.
# Gulkana has lots of nunataks, so there are lots of segments.
# 
# You'll also notice that the mesh we get is very fine.
# Using a resolution that high might not be necessary to get a result that's accurate enough and it will definitely be more expensive.
# Depending on your use case, it might be worth doing some preliminary coarsening of the initial geometry.
