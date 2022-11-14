# Original source code modified to add prediction batching support by Invitae in 2021.
# Modifications copyright (c) 2021 Invitae Corporation.

import logging
import shelve
import pysam
import collections
import os
import gc

from spliceai.utils import get_alt_gene_delta_score, is_record_valid, get_seq, \
    is_location_predictable, get_cov, get_wid, is_valid_alt_record, encode_seqs, create_unhandled_delta_score

logger = logging.getLogger(__name__)


## CUSTOM DATA TYPES
SequenceType_REF = 0
SequenceType_ALT = 1

BatchLookupIndex = collections.namedtuple(
    #                    ref/alt       size        batch for this size    index in current batch for this size
    'BatchLookupIndex', 'sequence_type tensor_size batch_ix batch_index'
)

PreparedVCFRecord = collections.namedtuple(
    'PreparedVCFRecord', 'vcf_idx gene_info locations'
)


## routine to create the batches for prediction.
def prepare_batches(ann, input_data,prediction_batch_size, prediction_queue,tmpdir,dist):
    # create the parser object
    vcf_reader = VCFReader(ann=ann, input_data=input_data, prediction_batch_size=prediction_batch_size, prediction_queue=prediction_queue,tmpdir=tmpdir,dist=dist)
    # parse records
    vcf_reader.add_records()
    # finalize last batches
    vcf_reader.finish()
    # close the shelf.
    vcf_reader.shelf_records.close()
    # stats 
    logger.info("Read {} vcf records, queued {} predictions".format(vcf_reader.total_vcf_records, vcf_reader.total_predictions))


## get tensorflow predictions using batch-based submissions
def get_preds(ann, x, batch_size=32):
    logger.debug('Running get_preds with matrix size: {}'.format(x.shape))
    try:
        predictions = [ann.models[m].predict(x, batch_size=batch_size, verbose=0) for m in range(5)]
        
    except Exception as e:
        # try a smaller batch (less efficient, but lower on memory). if it crashes again : it raises.
        logger.warning("TF.predict failed ({}).Retrying with smaller batch size".format(e))
        predictions = [ann.models[m].predict(x, batch_size=4, verbose=0) for m in range(5)]
    # garbage collection to prevent memory overflow... 
    gc.collect()
    return predictions
    


# Heavily based on utils.get_delta_scores but only handles the validation and encoding
# of the record, but doesn't do any of the prediction or post-processing steps
def encode_batch_records(record, ann, dist_var, gene_info):
    cov = get_cov(dist_var)
    wid = get_wid(cov)
    # If the record is not going to get a prediction, return this empty encoding
    empty_encoding = ([], [])

    if not is_record_valid(record):
        return empty_encoding

    seq = get_seq(record, ann, wid)
    if not seq:
        return empty_encoding

    if not is_location_predictable(record, seq, wid, dist_var):
        return empty_encoding

    all_x_ref = []
    all_x_alt = []
    for alt_ix in range(len(record.alts)):
        for gene_ix in range(len(gene_info.idxs)):

            if not is_valid_alt_record(record, alt_ix):
                continue

            x_ref, x_alt = encode_seqs(record=record,
                                       seq=seq,
                                       ann=ann,
                                       gene_info=gene_info,
                                       gene_ix=gene_ix,
                                       alt_ix=alt_ix,
                                       wid=wid)

            all_x_ref.append(x_ref)
            all_x_alt.append(x_alt)

    return all_x_ref, all_x_alt


# Heavily based on utils.get_delta_scores but only handles the post-processing steps after
# the models have made the predictions
def extract_delta_scores(
    all_y_ref, all_y_alt, record, ann, dist_var, mask, gene_info
):
    cov = get_cov(dist_var)
    delta_scores = []
    pred_ix = 0
    for alt_ix in range(len(record.alts)):
        for gene_ix in range(len(gene_info.idxs)):

            
            # Pull prediction out of batch
            try:
                y_ref = all_y_ref[pred_ix]
                y_alt = all_y_alt[pred_ix]
            except IndexError:
                logger.warn("No data for record below, alt_ix {} : gene_ix {} : pred_ix {}".format(alt_ix, gene_ix,pred_ix))
                logger.warn(record)
                continue
            except Exception as e:
                logger.error("Predction error: {}".format(e))
                logger.error(record)
                raise e

            # No prediction here
            if y_ref is None or y_alt is None:
                continue

            if not is_valid_alt_record(record, alt_ix):
                continue

            if len(record.ref) > 1 and len(record.alts[alt_ix]) > 1:
                pred_ix += 1
                delta_score = create_unhandled_delta_score(record.alts[alt_ix], gene_info.genes[gene_ix])
                delta_scores.append(delta_score)
                continue

            if pred_ix >= len(all_y_ref) or pred_ix >= len(all_y_alt):
                raise LookupError(
                    'Prediction index {} does not exist in prediction matrices: ref({}) alt({})'.format(
                        pred_ix, len(all_y_ref), len(all_y_alt)
                    )
                )

            delta_score = get_alt_gene_delta_score(record=record,
                                                   ann=ann,
                                                   alt_ix=alt_ix,
                                                   gene_ix=gene_ix,
                                                   y_ref=y_ref,
                                                   y_alt=y_alt,
                                                   cov=cov,
                                                   gene_info=gene_info,
                                                   mask=mask)
            delta_scores.append(delta_score)

            pred_ix += 1

    return delta_scores



# class to parse input and prep batches
class VCFReader:
    def __init__(self, ann, input_data, prediction_batch_size, prediction_queue,tmpdir,dist):
        self.ann = ann
        # This is the maximum number of predictions to parse/encode/predict at a time
        self.prediction_batch_size = prediction_batch_size
        # the vcf file
        self.input_data = input_data
        # window to consider
        self.dist = dist
        # Batch vars
        self.batches = {}
        #self.prepared_vcf_records = []

        # Counts
        self.total_predictions = 0
        self.total_vcf_records = 0
        self.batch_counters = {}
        
        # the queue
        self.prediction_queue = prediction_queue

        # shelves to track data. 
        self.tmpdir = tmpdir 
        # track records to have order correct
        self.shelf_records = shelve.open(os.path.join(self.tmpdir.name,"spliceai_records.shelf"))



    def add_records(self):
        
        try:
            vcf = pysam.VariantFile(self.input_data)
        except (IOError, ValueError) as e:
            logging.error('{}'.format(e))
            raise(e)
        for record in vcf:
            try:
                self.add_record(record)
            except Exception as e:
                raise(e)
        vcf.close()
        

    def add_record(self, record):
        """
        Adds a record to a batch. It'll capture the gene information for the record and
        save it for later to avoid looking it up again, then it'll encode ref and alt from
        the VCF record and place the encoded values into lists of matching sizes. Once the
        encoded values are added, a BatchLookupIndex is created so that after the predictions
        are made, it knows where to look up the corresponding prediction for the vcf record.

        Once the batch size hits it's capacity, it'll process all the predictions for the
        encoded batch.
        """

        self.total_vcf_records += 1
        # Collect gene information for this record
        gene_info = self.ann.get_name_and_strand(record.chrom, record.pos)

        # Keep track of how many predictions we're going to make
        prediction_count = len(record.alts) * len(gene_info.genes)
        self.total_predictions += prediction_count

        # Collect lists of encoded ref/alt sequences
        x_ref, x_alt = encode_batch_records(record, self.ann, self.dist, gene_info)

        # List of BatchLookupIndex's so we know how to lookup predictions for records from
        # the batches
        batch_lookup_indexes = []

        # Process the encodings into batches
        for var_type, encoded_seq in zip((SequenceType_REF, SequenceType_ALT), (x_ref, x_alt)):

            if len(encoded_seq) == 0:
                # Add BatchLookupIndex with zeros so when the batch collects the outputs
                # it knows that there is no prediction for this record
                batch_lookup_indexes.append(BatchLookupIndex(var_type, 0, 0, 0))
                continue

            # Iterate over the encoded sequence and drop into the correct batch by size and
            # create an index to use to pull out the result after batch is processed
            for row in encoded_seq:
                # Extract the size of the sequence that was encoded to build a batch from
                tensor_size = row.shape[1]

                # Create batch for this size
                if tensor_size not in self.batches:
                    self.batches[tensor_size] = []
                    self.batch_counters[tensor_size] = 0

                # Add encoded record to batch 'n' for tensor_size 
                self.batches[tensor_size].append(row)

                # Get the index of the record we just added in the batch
                cur_batch_record_ix = len(self.batches[tensor_size]) - 1

                # Store a reference so we can pull out the prediction for this item from the batches
                batch_lookup_indexes.append(BatchLookupIndex(var_type, tensor_size, self.batch_counters[tensor_size] , cur_batch_record_ix))

        # Save the batch locations for this record on the composite object
        prepared_record = PreparedVCFRecord(
            vcf_idx=self.total_vcf_records, gene_info=gene_info, locations=batch_lookup_indexes
        )
        #  add to shelf by vcf_idx
        self.shelf_records[str(self.total_vcf_records)] = prepared_record

        # If we're reached our threshold for the max items to process, then process the batch
        for tensor_size in self.batch_counters:
            if len(self.batches[tensor_size]) >= self.prediction_batch_size:
                logger.debug("Batch {} full. Adding to queue".format(tensor_size))
                queue_item = {'tensor_size': tensor_size, 'batch_ix': self.batch_counters[tensor_size], 'data' : self.batches[tensor_size]}
                self.prediction_queue.put(queue_item)
                # reset
                self.batches[tensor_size] = []
                self.batch_counters[tensor_size] += 10

                #self._process_batch(tensor_size)
        


    def finish(self):
        """
        Method to process all the remaining items that have been added to the batches.
        """
        #if len(self.prepared_vcf_records) > 0:
        #    self._process_batch()
        logger.debug("Queueing remaining batches")
        for tensor_size in self.batch_counters:
                if len(self.batches[tensor_size] ) > 0:
                    queue_item = {'tensor_size': tensor_size, 'batch_ix': self.batch_counters[tensor_size], 'data' : self.batches[tensor_size]}
                    self.prediction_queue.put(queue_item)
                    # clear
                    self.batches[tensor_size] = []
        # all done : 
        self.prediction_queue.put(None)