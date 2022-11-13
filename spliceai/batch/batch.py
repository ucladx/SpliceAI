# Original source code modified to add prediction batching support by Invitae in 2021.
# Modifications copyright (c) 2021 Invitae Corporation.

import collections
import logging
import time
import shelve
import tempfile
import numpy as np
import os

from spliceai.batch.batch_utils import extract_delta_scores, get_preds, encode_batch_records

logger = logging.getLogger(__name__)

SequenceType_REF = 0
SequenceType_ALT = 1


BatchLookupIndex = collections.namedtuple(
    #                    ref/alt       size        batch for this size    index in current batch for this size
    'BatchLookupIndex', 'sequence_type tensor_size batch_ix batch_index'
)

PreparedVCFRecord = collections.namedtuple(
    'PreparedVCFRecord', 'vcf_idx gene_info locations'
)


class VCFPredictionBatch:
    def __init__(self, ann, output, dist, mask, prediction_batch_size, tensorflow_batch_size):
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
        self.batch_predictions = 0
        self.total_predictions = 0
        self.total_vcf_records = 0
        self.batch_counters = {}
        

        # shelves to track data. 
        self.tmpdir = tempfile.TemporaryDirectory()
        # store batches of predictions using 'tensor_size|batch_idx' as key.
        self.shelf_preds = shelve.open(os.path.join(self.tmpdir.name,"spliceai_preds.shelf"))
        # track records to have order correct
        self.shelf_records = shelve.open(os.path.join(self.tmpdir.name,"spliceai_records.shelf"))


    def _process_batch(self,tensor_size):
        start = time.time()
        # get last batch for this tensor_size
        batch_ix = self.batch_counters[tensor_size]
        batch = self.batches[tensor_size]
        # Sanity check dump of batch sizes
        logger.debug('Tensor size : {} : batch_ix {} : nr.entries : {}'.format(tensor_size, batch_ix , len(batch)))

        # Convert list of encodings into a proper sized numpy matrix
        prediction_batch = np.concatenate(batch, axis=0)
        # Run predictions && add to shelf.
        self.shelf_preds["{}|{}".format(tensor_size,batch_ix)] = np.mean(
            get_preds(self.ann, prediction_batch, self.prediction_batch_size), axis=0
        )

        # clear the batch.
        self.batches[tensor_size] = []
        # initialize the next batch_ix
        self.batch_counters[tensor_size] += 1
        
        logger.debug('Predictions: {}, VCF Records: {}'.format(self.total_predictions, self.total_vcf_records))
        duration = time.time() - start
        preds_per_sec = len(batch) / duration
        preds_per_hour = preds_per_sec * 60 * 60
        logger.debug('Finished in {:0.2f}s, per sec: {:0.2f}, per hour: {:0.2f}'.format(duration,
                                                                                        preds_per_sec,
                                                                                        preds_per_hour))

    # wrapper to write out all shelved variants
    def write_records(self, vcf):
        line_idx = 0
        for record in vcf:
            line_idx += 1  
            # get prepared record by line_idx
            prepared_record = self.shelf_records[str(line_idx)]
            #record = prepared_record.vcf_record
            gene_info = prepared_record.gene_info
            record_predictions = 0

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
                batch = self.shelf_preds["{}|{}".format(location.tensor_size,location.batch_ix)] # batch_preds[location.tensor_size]
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
                record_predictions += len(delta_scores)

            self.output.write(record)
    

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
        self.batch_predictions += prediction_count
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
                logger.debug("Batch {} full. Processing".format(tensor_size))
                self._process_batch(tensor_size)
        


    def finish(self):
        """
        Method to process all the remaining items that have been added to the batches.
        """
        #if len(self.prepared_vcf_records) > 0:
        #    self._process_batch()
        logger.debug("Processing remaining batches")
        for tensor_size in self.batch_counters:
                if len(self.batches[tensor_size] ) > 0:
                    self._process_batch(tensor_size)
