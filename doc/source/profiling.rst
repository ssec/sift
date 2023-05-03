SIFT Profiling
==================

SIFT can be started with a custom Heap Profiler, which is using the Python
tracemalloc module underneath. When done so it creates a directory
``<YYYYddmm_HH-MM>_uwsift_heap_profile`` in the current working directory and stores
memory usage snapshot data into it. The command to take a snapshot every 2.0
seconds is as follows::

  python -m uwsift --profile-heap 2.0

Afterwards the snapshorts must be combined::

  python ./uwsift/util/heap_analyzer.py --combine ./combined_stats.prof --snapshot-dir <<YYYYddmm_HH-MM>_uwsift_heap_profile>

This creates the file ``combined_stats.prof``, which can be visualized with
matplotlib for analysis::

  python ./uwsift/util/heap_analyzer.py --load ./combined_stats.prof --plot
