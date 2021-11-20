import os
import time


def file_write_log(args, *texts):
    # print('')
    if args is None:
        args.local_rank = 0
        args.output_dir = '/.output'
        args.train_log_nam = 'text'

    if args.local_rank in [-1, 0]:
        with open(os.path.join(args.output_dir, args.train_log_name), 'a') as f:
            f.write('\n')
            localtime = time.asctime(time.localtime(time.time()))
            f.write(str(localtime))
            f.write('\n')
            for test in texts:
                f.write(test)
