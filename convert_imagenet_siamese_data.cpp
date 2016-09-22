// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");

bool ReadTwoImageToOneDatum(const std::string filename1, const int label1, const std::string filename2,
		const int label2, const int resize_height, const int resize_width, const bool is_color, Datum* datum);

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 5) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;

  std::ifstream infile(argv[2]);
  CHECK(infile) << "Unable to open file "<< argv[2];
  std::vector<std::pair<std::string, int> > lines;
  std::string filename;
  int label;
  while (infile >> filename >> label) {
    lines.push_back(std::make_pair(filename, label));
  }
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[4], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  std::string root_folder(argv[1]);
  Datum datum;
  int count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  int data_size = 0;
  bool data_size_initialized = false;
  
  //open image pairs data
  std::ifstream pair_file(argv[3]);
  CHECK(pair_file) << "Unable to open file " << argv[3];
  
  std::string c1,c2,s1,s2;
  //load image data
  for (int line_id = 0; line_id < lines.size(); ++line_id) {
	//read image pairs
	pair_file >> c1 >> c2 >> s1 >> s2;
	int line_id1 = atoi(c1.c_str());
	int line_id2 = atoi(c2.c_str()); 
    //load data
    bool status;
    std::string enc = encode_type;
    if (encoded && !enc.size()) {
      // Guess the encoding type from the file name
      string fn = lines[line_id1].first;
      size_t p = fn.rfind('.');
      if ( p == fn.npos )
        LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
      enc = fn.substr(p);
      std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
    }
    /*
    status = ReadImageToDatum(root_folder + lines[line_id].first,
        lines[line_id].second, resize_height, resize_width, is_color,
        enc, &datum);
    */
    LOG(INFO) << "Image Pairs: " << lines[line_id1].first << " & " << lines[line_id2].first;
    status = ReadTwoImageToOneDatum(root_folder + lines[line_id1].first, lines[line_id1].second,
        		root_folder + lines[line_id2].first, lines[line_id2].second,
        		resize_height, resize_width, is_color, &datum);

    if (status == false) continue;
    if (check_size) {
      if (!data_size_initialized) {
        data_size = datum.channels() * datum.height() * datum.width();
        data_size_initialized = true;
      } else {
        const std::string& data = datum.data();
        CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
            << data.size();
      }
    }
    // sequential
    //int length = snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id,
    //    lines[line_id].first.c_str());
    
    int length = snprintf(key_cstr, kMaxKeyLength, "%08d", line_id);
    // Put in db
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(string(key_cstr, length), out);

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
  return 0;
}

bool ReadTwoImageToOneDatum(const std::string filename1, const int label1, const std::string filename2,
		const int label2, const int resize_height, const int resize_width, const bool is_color, Datum* datum){

	    cv::Mat cv_img1 = ReadImageToCVMat(filename1, resize_height, resize_width, is_color);
	    cv::Mat cv_img2 = ReadImageToCVMat(filename2, resize_height, resize_width, is_color);
	    //CVMatToDatum
	    CHECK(cv_img1.depth() == CV_8U) << "Image data type must be unsigned byte";
	    CHECK(cv_img2.depth() == CV_8U) << "Image data type must be unsigned byte";
	    CHECK(cv_img1.channels() == cv_img2.channels()) << "The two image channels is mismatch";
	    datum->set_channels(cv_img1.channels() + cv_img2.channels());
	    datum->set_height(cv_img1.rows);
	    datum->set_width(cv_img1.cols);
	    datum->clear_data();
	    datum->clear_float_data();
	    datum->set_encoded(false);LOG(INFO)<<"2";
	    int datum_channels = datum->channels();
	    int datum_height = datum->height();
	    int datum_width = datum->width();
	    int datum_size = datum_channels * datum_height * datum_width;
	    std::string buffer(datum_size, ' ');
	    int img1_size = cv_img1.channels() * cv_img1.rows * cv_img1.cols;
	   // LOG(INFO)<<"data arguments: "<<datum_height<<"," << datum_width << "," << datum_channels;
	    for (int h = 0; h < datum_height; ++h) {
	        const uchar* ptr1 = cv_img1.ptr<uchar>(h);
	        const uchar* ptr2 = cv_img2.ptr<uchar>(h);
	        int img_index = 0;
	        for (int w = 0; w < datum_width; ++w) {
	          for (int c = 0; c < cv_img1.channels(); ++c) {
	            int datum_index = (c * datum_height + h) * datum_width + w;
	            buffer[datum_index] = static_cast<char>(ptr1[img_index]);
	            buffer[datum_index + img1_size] = static_cast<char>(ptr2[img_index]);
	            img_index++;
	          }
	        }
	      }
	     datum->set_data(buffer);
	     //set label
	     if (label1 == label2){
	    	 datum->set_label(1);
	     }
	     else{
	    	 datum->set_label(0);
	     }
	     return true;
}
