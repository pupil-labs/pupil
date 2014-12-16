import sys,os,platform
from ctypes import c_bool
import time
import logging
import shelve
import curses
from textwrap import dedent
import argparse

from multiprocessing import Process,cpu_count
from multiprocessing.sharedctypes import Value



# Make all pupil shared_modules available to this Python session.
pupil_base_dir = os.path.abspath(__file__).rsplit('pupil_src', 1)[0]
sys.path.append(os.path.join(pupil_base_dir, 'pupil_src', 'shared_modules'))

from methods import Temp
from player_methods import is_pupil_rec_dir,patch_meta_info

from exporter import export
logger = logging.getLogger()
logger.setLevel(logging.WARNING)

ch = logging.StreamHandler()
formatter = logging.Formatter('Batch Export [%(levelname)s] %(name)s : %(message)s')
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(ch)
# create logger for the context of this function
logger = logging.getLogger(__name__)

#set level of local logger
logger.setLevel(logging.INFO)


def color_lookup(color_name):
    colors = {"green": (0/255.,0/255.,255/255.,125/255.),
            "green": (0/255.,255/255.,0/255.,125/255.),
            "red": (255/255.,0/255.,0/255.,125/255.),
             }
    # return tuple of value for each color name
    return colors[color_name]


def get_recording_dirs(data_dir):
    '''
        You can supply a data folder or any folder
        - all folders within will be checked for necessary files
        - in order to make a visualization
    '''
    filtered_recording_dirs = []
    if is_pupil_rec_dir(data_dir):
        filtered_recording_dirs.append(data_dir)
    for root,dirs,files in os.walk(data_dir):
        filtered_recording_dirs += [os.path.join(root,d) for d in dirs if not d.startswith(".") and is_pupil_rec_dir(os.path.join(root,d))]
    logger.debug("Filtered Recording Dirs: %s" %filtered_recording_dirs)
    return filtered_recording_dirs


def show_progess(jobs):
    no_jobs = len(jobs)
    width = 120
    full = width/no_jobs
    string = ""
    for j in jobs:
        try:
            p = int(width*j.current_frame.value/float(j.frames_to_export.value*no_jobs) )
        except:
            p = 0
        string += '['+ p*"|"+(full-p)*"-" + "]"
    sys.stdout.write("\r"+string)
    sys.stdout.flush()


def main():
    """Batch process recordings to produce visualizations
    Using simple_circle as the default visualizations
    Steps:
        - User Supplies: Directory that contains many recording(s) dirs or just one recordings dir
        - We walk the user supplied directory to get all data folders
        - Data is the list we feed to our multiprocessed
        - Error check -- do we have required files in each dir?: world.avi, gaze_positions.npy, timestamps.npy
        - Result: world_viz.avi within each original data folder
    """


    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
        description=dedent('''\
            ***************************************************
            Batch process recordings to produce visualizations
            The default visualization will use simple_circle

            Usage Example:
                python batch_export.py -d /path/to/folder-with-many-recordings -s ~/Pupil_Player/settings/user_settings -e ~/my_export_dir
            Arguments:
                -d : Specify a recording directory.
                     This could have one or many recordings contained within it.
                     We will recurse into the dir.
                -s : Specify path to Pupil Player user_settings file to use last used vizualization settings.
                -e : Specify export directory if you dont what the export saved within each recording dir.
                -c : If you dont use the player settings file, we use a default circle.
                     You can specify color to be used
                     Available options:
                        white, red, green
                -p : Export a 120 frame preview only.
            ***************************************************\
        '''))
    parser.add_argument('-d', '--rec-dir',required=True)
    parser.add_argument('-s', '--settings-file',default=False)
    parser.add_argument('-e', '--export-to-dir',default=False)
    parser.add_argument('-c', '--basic-color',default='red')
    parser.add_argument('-p', '--preview', action='store_true')

    if len(sys.argv)==1:
        print parser.description
        return

    args = parser.parse_args()
    # get the top level data folder from terminal argument

    data_dir = args.rec_dir

    if args.settings_file and os.path.isfile(args.settings_file):
        session_settings = shelve.open(os.path.splitext(args.settings_file)[0],protocol=2)
        #these are loaded based on user settings
        plugin_initializers = session_settings.get('plugins',[])
        session_settings.close()

        for initializer in plugin_initializers:
            name, var = initializer
            logger.debug("Loading plugin: %s with settings %s"%(name, var))
    else:
        session_settings = {}
        if args.basic_color:
            try:
                color = color_lookup(args.basic_color)
            except KeyError:
                logger.warning("Not a real color. Choosing red")
                color = color_lookup('red')
        else:
            logger.warning("No color selected. Choosing red")
            color = color_lookup('red')

        #load a simple cirlce plugin as default
        logger.debug("Loading default plugin: Vis_Circle with color: %s"%(color,))

        plugin_initializers = [("Vis_Circle",{'thickness':2,'color':color})]


    if args.export_to_dir:
        export_dir = args.export_to_dir
        if os.path.isdir(export_dir):
            logger.info("Exporting all vids to %s"%export_dir)
        else:
            logger.info("Exporting dir is not valid %s"%export_dir)
            return
    else:
        export_dir = None

    if args.preview:
        preview = True
    else:
        preview =  False


    g = Temp()
    g.app = "batch_export"

    recording_dirs = get_recording_dirs(data_dir)
    for r in recording_dirs:
        patch_meta_info(r)

    # start multiprocessing engine
    n_cpu = cpu_count()
    logger.info("Using a maximum of %s CPUs to process visualizations in parallel..." %n_cpu)


    jobs = []
    outfiles = set()
    for d in recording_dirs:
        j = Temp()
        logger.info("Adding new export: %s"%d)
        j.should_terminate = Value(c_bool,0)
        j.frames_to_export  = Value('i',0)
        j.current_frame = Value('i',0)
        j.data_dir = d
        j.start_frame= None
        if preview:
            j.end_frame = 30
        else:
            j.end_frame = None
        j.plugin_initializers = plugin_initializers[:]

        if export_dir:
            #make a unique name created from rec_session and dir name
            rec_session, rec_dir = d.rsplit(os.path.sep,2)[1:]
            out_name = rec_session+"_"+rec_dir+".avi"
            j.out_file_path = os.path.join(os.path.expanduser(export_dir),out_name)
            if j.out_file_path in outfiles:
                logger.error("This export setting would try to save %s at least twice pleace rename dirs to prevent this."%j.out_file_path)
                return
            outfiles.add(j.out_file_path)
            logger.info("Exporting to: %s"%j.out_file_path)

        else:
            j.out_file_path = None

        j.args = (j.should_terminate,j.frames_to_export,j.current_frame, j.data_dir,j.start_frame,j.end_frame,j.plugin_initializers,j.out_file_path)
        jobs.append(j)


    todo = jobs[:]
    workers = [Process(target=export,args=todo.pop(0).args) for i in range(min(len(todo),n_cpu))]
    for w in workers:
        w.start()

    working = True

    t = time.time()
    while working: #cannot use pool as it does not allow shared memory
        working = False
        for i in range(len(workers)):
            if workers[i].is_alive():
                working = True
            else:
                if todo:
                    workers[i] = Process(target=export,args=todo.pop(0).args)
                    workers[i].start()
                    working = True
        show_progess(jobs)
        time.sleep(.25)
    print "\n"

    #lets give some cpu performance feedback.
    total_frames = sum((j.frames_to_export.value for j in jobs))
    total_secs = time.time()-t
    logger.info("Export done. Exported %s frames in %s seconds. %s fps"%(total_frames,total_secs,total_frames/total_secs))


if __name__ == '__main__':
    main()
