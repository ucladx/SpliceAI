# Original source code modified to add prediction batching support by Invitae in 2021.
# Modifications copyright (c) 2021 Invitae Corporation.

# Invitae source code modified to improve GPU utilization
# Modifications made by Geert Vandeweyer (Antwerp University Hospital, Belgium)

import logging
import time
import shelve
import numpy as np
import os
import tensorflow as tf
import pickle

from spliceai.batch.batch_utils import extract_delta_scores, get_preds

logger = logging.getLogger(__name__)

SequenceType_REF = 0
SequenceType_ALT = 1


# Class to handle predictions
class VCFPredictionBatch:
    def __init__(self, ann, output, dist, mask, prediction_batch_size, tensorflow_batch_size,tmpdir):
        self.ann = ann
        self.output = output
        self.dist = dist
        self.mask = mask
        # This is the maximum number of predictions to parse/encode/predict at a time
        self.prediction_batch_size = prediction_batch_size
        # This is the size of the batch tensorflow will use to make the predictions
        self.tensorflow_batch_size = tensorflow_batch_size
        # track runtime
        self.start_time = time.time()
        # Batch vars
        self.batches = {}
        self.prepared_vcf_records = []

        # Counts
        self.total_predictions = 0
        self.total_vcf_records = 0
        self.batch_counters = {}
        
        

        # shelves to track data. 
        self.tmpdir = tmpdir 
        # store batches of predictions using 'tensor_size|batch_idx' as key.
        self.shelf_preds = shelve.open(os.path.join(self.tmpdir.name,"spliceai_preds.shelf"))
        
    # monitor the queue and submit incoming batches.
    def process_batches(self,prediction_queue):
        while True:
            item =prediction_queue.get()
            # reader submits None when all are queued.
            if item is None:
                break
            # load pickled object
            with open(os.path.join(self.tmpdir.name,item),'rb') as p:
                data = pickle.load(p)
            # remove from disk.
            os.unlink(os.path.join(self.tmpdir.name,item))
            self._process_batch(data['tensor_size'],data['batch_ix'], data['data'],data['length'])
            

    def _process_batch(self,tensor_size,batch_ix, prediction_batch,nr_preds):
        start = time.time()
        
        # Sanity check dump of batch sizes
        logger.debug('Tensor size : {} : batch_ix {} : nr.entries : {}'.format(tensor_size, batch_ix , nr_preds))

        # Run predictions && add to shelf.
        self.shelf_preds["{}|{}".format(tensor_size,batch_ix)] = np.mean(
            get_preds(self.ann, prediction_batch, self.tensorflow_batch_size), axis=0
        )
        
        # status
        duration = time.time() - start
        preds_per_sec = nr_preds / duration
        preds_per_hour = preds_per_sec * 60 * 60
        logger.debug('Finished in {:0.2f}s, per sec: {:0.2f}, per hour: {:0.2f}'.format(duration,
                                                                                        preds_per_sec,
                                                                                        preds_per_hour))

    # wrapper to write out all shelved variants
    def write_records(self, vcf):
        # open the shelf with records:
        shelf_records = shelve.open(os.path.join(self.tmpdir.name,"spliceai_records.shelf"))
        # parse vcf
        line_idx = 0
        batch = []
        last_batch_key = ''
        for record in vcf:
            line_idx += 1  
            # get prepared record by line_idx
            prepared_record = shelf_records[str(line_idx)]
            gene_info = prepared_record.gene_info
            # (REF + #ALT ) * #genes   (* 5 models)
            self.total_predictions += (1 + len(record.alts)) * len(gene_info.genes)
            
            all_y_ref = []
            all_y_alt = []

            # Each prediction in the batch is located and put into the correct y
            for location in prepared_record.locations:
                # No prediction here
                if location.tensor_size == 0:
                    if location.sequence_type == SequenceType_REF:
                        all_y_ref.append(None)
                    else:
                        all_y_alt.append(None)
                    continue
                
                # Extract the prediction from the batch into a list of predictions for this record
                # recycle the batch variable if key is the same.
                if not last_batch_key == "{}|{}".format(location.tensor_size,location.batch_ix):
                    last_batch_key = "{}|{}".format(location.tensor_size,location.batch_ix)
                    batch = self.shelf_preds[last_batch_key] # batch_preds[location.tensor_size]
                    
                if location.sequence_type == SequenceType_REF:
                    all_y_ref.append(batch[[location.batch_index], :, :])
                else:
                    all_y_alt.append(batch[[location.batch_index], :, :])
            delta_scores = extract_delta_scores(
                all_y_ref=all_y_ref,
                all_y_alt=all_y_alt,
                record=record,
                ann=self.ann,
                dist_var=self.dist,
                mask=self.mask,
                gene_info=gene_info,
            )

            # If there are predictions, write them to the VCF INFO section
            if len(delta_scores) > 0:
                record.info['SpliceAI'] = delta_scores

            self.output_data.write(record)
        # close shelf again
        self.total_vcf_records = line_idx
        shelf_records.close()
    

