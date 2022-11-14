# Original source code modified to add prediction batching support by Invitae in 2021.
# Modifications copyright (c) 2021 Invitae Corporation.

import sys
import argparse
import logging
import pysam
import time
import tempfile
from multiprocessing import Process,Queue

from spliceai.batch.batch import VCFPredictionBatch
from spliceai.batch.batch_utils import prepare_batches
from spliceai.utils import Annotator, get_delta_scores

try:
    from sys.stdin import buffer as std_in
    from sys.stdout import buffer as std_out
except ImportError:
    from sys import stdin as std_in
    from sys import stdout as std_out


def get_options():

    parser = argparse.ArgumentParser(description='Version: 1.3.1')
    parser.add_argument('-I', metavar='input', nargs='?', default=std_in,
                        help='path to the input VCF file, defaults to standard in')
    parser.add_argument('-O', metavar='output', nargs='?', default=std_out,
                        help='path to the output VCF file, defaults to standard out')
    parser.add_argument('-R', metavar='reference', required=True,
                        help='path to the reference genome fasta file')
    parser.add_argument('-A', metavar='annotation', required=True,
                        help='"grch37" (GENCODE V24lift37 canonical annotation file in '
                             'package), "grch38" (GENCODE V24 canonical annotation file in '
                             'package), or path to a similar custom gene annotation file')
    parser.add_argument('-D', metavar='distance', nargs='?', default=50,
                        type=int, choices=range(0, 5000),
                        help='maximum distance between the variant and gained/lost splice '
                             'site, defaults to 50')
    parser.add_argument('-M', metavar='mask', nargs='?', default=0,
                        type=int, choices=[0, 1],
                        help='mask scores representing annotated acceptor/donor gain and '
                             'unannotated acceptor/donor loss, defaults to 0')
    parser.add_argument('-B', '--prediction-batch-size', metavar='prediction_batch_size', default=1, type=int,
                        help='number of predictions to process at a time, note a single vcf record '
                             'may have multiple predictions for overlapping genes and multiple alts')
    parser.add_argument('-T', '--tensorflow-batch-size', metavar='tensorflow_batch_size', type=int,
                        help='tensorflow batch size for model predictions')
    parser.add_argument('-V', '--verbose', action='store_true', help='enables verbose logging')
    args = parser.parse_args()

    return args


def main():
    args = get_options()

    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(name)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=loglevel,
    )
    
    if None in [args.I, args.O, args.D, args.M]:
        logging.error('Usage: spliceai [-h] [-I [input]] [-O [output]] -R reference -A annotation '
                      '[-D [distance]] [-M [mask]] [-B [prediction_batch_size]] [-T [tensorflow_batch_size]]')
        exit()

    # Default the tensorflow batch size to the prediction_batch_size if it's not supplied in the args
    tensorflow_batch_size = args.tensorflow_batch_size if args.tensorflow_batch_size else args.prediction_batch_size

    # load annotation data:
    ann = Annotator(args.R, args.A)
    ## revised code for batched analysis
    if args.prediction_batch_size > 1:
        run_spliceai_batched(input_data=args.I, output_data=args.O, reference=args.R,
             ann=ann, distance=args.D, mask=args.M,
             prediction_batch_size=args.prediction_batch_size,
             tensorflow_batch_size=tensorflow_batch_size)
    else: # run original code:
        run_spliceai(input_data=args.I, output_data=args.O, ann=ann, distance=args.D, mask=args.M)

## revised logic to allow batched tensorflow analysis
def run_spliceai_batched(input_data, output_data, reference, ann, distance, mask, prediction_batch_size,
                 tensorflow_batch_size):
    
    ## mk a temp directory 
    tmpdir = tempfile.TemporaryDirectory()
    # initialize the prediction object
    batch = VCFPredictionBatch(
            ann=ann,
            output=output_data,
            dist=distance,
            mask=mask,
            prediction_batch_size=prediction_batch_size,
            tensorflow_batch_size=tensorflow_batch_size,
            tmpdir = tmpdir
        )
    
    # creates a queue with max 10 ready-to-go batches in it.
    # starts processing & filling the queue.
    prediction_queue = Queue(maxsize=10)
    reader = Process(target=prepare_batches, kwargs={'ann':ann, 
                                                     'input_data':input_data, 
                                                     'prediction_batch_size':prediction_batch_size, 
                                                     'prediction_queue': prediction_queue,
                                                     'tmpdir':tmpdir,
                                                     'dist' : distance
                                                    })
    reader.start()    
    
    # Process the queue.
    batch.process_batches(prediction_queue)

    # join the reader process.
    reader.join()

    # stats without writing phase
    prediction_duration = time.time() - batch.start_time
    
    # write results.
    # Iterate over original list of vcf records again, reconstructing record with annotations from shelved data
    logging.debug("Writing output file")
    vcf = pysam.VariantFile(input_data)
    # have to update header again
    header = vcf.header
    header.add_line('##INFO=<ID=SpliceAI,Number=.,Type=String,Description="SpliceAIv1.3.1 variant '
                'annotation. These include delta scores (DS) and delta positions (DP) for '
                'acceptor gain (AG), acceptor loss (AL), donor gain (DG), and donor loss (DL). '
                'Format: ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL">')
    try:
        batch.output_data = pysam.VariantFile(output_data, mode='w', header=header)
        
    except (IOError, ValueError) as e:
        logging.error('{}'.format(e))
        exit()

    batch.write_records(vcf)
    # close shelf
    batch.shelf_preds.close()
    # close vcf
    vcf.close()
    batch.output_data.close()


    ## stats
    overall_duration = time.time() - batch.start_time
    preds_per_sec = batch.total_predictions / prediction_duration
    preds_per_hour = preds_per_sec * 60 * 60
    logging.info("Analysis Finished. Statistics:")
    logging.info("Total RunTime: {:0.2f}s".format(overall_duration))
    logging.info("Prediction RunTime: {:0.2f}s".format(prediction_duration))
    logging.info("Processed Records: {}".format(batch.total_vcf_records))
    logging.info("Processed Predictions: {}".format(batch.total_predictions))
    logging.info("Overall performance : {:0.2f} predictions/sec ; {:0.2f} predictions/hour".format(preds_per_sec, preds_per_hour))


# original flow : record by record reading/predict/write
def run_spliceai(input_data, output_data, ann, distance, mask):

    try:
        vcf = pysam.VariantFile(input_data)
    except (IOError, ValueError) as e:
        logging.error('{}'.format(e))
        exit()

    header = vcf.header
    header.add_line('##INFO=<ID=SpliceAI,Number=.,Type=String,Description="SpliceAIv1.3.1 variant '
                    'annotation. These include delta scores (DS) and delta positions (DP) for '
                    'acceptor gain (AG), acceptor loss (AL), donor gain (DG), and donor loss (DL). '
                    'Format: ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL">')

    try:
        output_data = pysam.VariantFile(output_data, mode='w', header=header)
    except (IOError, ValueError) as e:
        logging.error('{}'.format(e))
        exit()

    for record in vcf:
            scores = get_delta_scores(record, ann, distance, mask)
            if len(scores) > 0:
                record.info['SpliceAI'] = scores
            output_data.write(record)
   
    # close VCF
    vcf.close()
    output_data.close()

if __name__ == '__main__':
    main()
