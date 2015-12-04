//
// This script converts the CIFAR dataset to the leveldb format used
// by caffe to perform classification.
// Usage:
//    convert_cifar_data input_folder output_db_file
// The CIFAR dataset could be downloaded at
//    http://www.cs.toronto.edu/~kriz/cifar.html

#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "boost/scoped_ptr.hpp"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "stdint.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

using caffe::Datum;
using boost::scoped_ptr;
using std::string;
namespace db = caffe::db;

const int kCIFARSize = 32;
const int kCIFARImageNBytes = 3072;
const int kCIFARBatchSize = 10000;

void read_image(std::ifstream* file, int* label, char* buffer, int index) {
  char label_char;
  file->seekg((kCIFARImageNBytes + 1)* index);
  file->read(&label_char, 1);
  *label = label_char;
  file->read(buffer, kCIFARImageNBytes);
}

void convert_dataset(const string& input_folder, const string& output_folder,
    const string& db_type, const string& train_pairs, const string& test_pairs) {
	
  const int kMaxKeyLength = 10;
  char key[kMaxKeyLength];
  
  scoped_ptr<db::DB> train_db(db::GetDB(db_type));
  train_db->Open(output_folder + "/cifar10_train_" + db_type, db::NEW);
  scoped_ptr<db::Transaction> txn(train_db->NewTransaction());
  // Data buffer
  int label1, label2;
  char* str_buffer = new char[kCIFARImageNBytes * 2];
  Datum datum;
  datum.set_channels(6);
  datum.set_height(kCIFARSize);
  datum.set_width(kCIFARSize);

  //open pairs file
  std::ifstream pair_train(train_pairs.c_str());//open train pairs file
  std::ifstream pair_test(test_pairs.c_str());//open test pairs file
  CHECK(pair_train) << "Unable to open file: " << train_pairs;
  CHECK(pair_test) << "Unable to open file: " << test_pairs;
  string c1, c2, s1, s2;

  LOG(INFO) << "Writing Training data";
  string filename = input_folder + "/data_batch.bin";
  std::ifstream train_data(filename.c_str(), std::ios::in | std::ios::binary);
  CHECK(train_data) << "Unable to open train file " << filename;
  for (int itemid = 0; itemid < kCIFARBatchSize * 5; ++itemid) {
      pair_train >> c1 >> c2 >> s1 >> s2;
      int i = atoi(c1.c_str());
      int j = atoi(c2.c_str());
      read_image(&train_data, &label1, str_buffer, i);
      read_image(&train_data, &label2, str_buffer + kCIFARImageNBytes, j);
      CHECK(label1 == atoi(s1.c_str())) << "The image index is mismatch.";
      CHECK(label2 == atoi(s2.c_str())) << "The image index is mismatch.";
      if (label1 == label2){
    	  datum.set_label(1);
      }
      else{
    	  datum.set_label(0);
      }
      datum.set_data(str_buffer, kCIFARImageNBytes * 2);
      int length = snprintf(key, kMaxKeyLength, "%05d", itemid);
      string out;
      CHECK(datum.SerializeToString(&out));
      txn->Put(string(key, length), out);
    }
  txn->Commit();
  train_db->Close();
  train_data.close();
  
  LOG(INFO) << "Writing Testing data";
  scoped_ptr<db::DB> test_db(db::GetDB(db_type));
  test_db->Open(output_folder + "/cifar10_test_" + db_type, db::NEW);
  txn.reset(test_db->NewTransaction());
  // Open files
  std::ifstream test_data((input_folder + "/test_batch.bin").c_str(),
      std::ios::in | std::ios::binary);
  CHECK(test_data) << "Unable to open test file.";
  for (int itemid = 0; itemid < kCIFARBatchSize; ++itemid) {
	pair_test >> c1 >> c2 >> s1 >> s2;
	int i = atoi(c1.c_str());
	int j = atoi(c2.c_str());
    read_image(&test_data, &label1, str_buffer, i);
    read_image(&test_data, &label2, str_buffer + kCIFARImageNBytes, j);
    CHECK(label1 == atoi(s1.c_str())) << "The image index is mismatch.";
    CHECK(label2 == atoi(s2.c_str())) << "The image index is mismatch.";
    if (label1 == label2){
          datum.set_label(1);
       }
    else{
       	  datum.set_label(0);
       }
    datum.set_data(str_buffer, kCIFARImageNBytes * 2);
    int length = snprintf(key, kMaxKeyLength, "%05d", itemid);
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(string(key, length), out);
  }
  txn->Commit();
  test_db->Close();
  test_data.close();
  
  pair_train.close();
  pair_test.close();
  delete []str_buffer;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    printf("This script converts the CIFAR dataset to the leveldb format used\n"
           "by caffe to perform classification.\n"
           "Usage:\n"
           "    convert_cifar_data input_folder output_folder db_type\n"
           "Where the input folder should contain the binary batch files.\n"
           "The CIFAR dataset could be downloaded at\n"
           "    http://www.cs.toronto.edu/~kriz/cifar.html\n"
           "You should gunzip them after downloading.\n");
  } else {
    google::InitGoogleLogging(argv[0]);
    convert_dataset(string(argv[1]), string(argv[2]), string(argv[3]), string(argv[4]), string(argv[5]));
  }
  return 0;
}
