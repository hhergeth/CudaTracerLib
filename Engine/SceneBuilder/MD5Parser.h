#pragma once

#include <string>
#include <stdexcept>
#include <vector>
#include <fstream>
#include <MathTypes.h>

enum TOKEN { TOKEN_KEYWORD,
               TOKEN_INT,
               TOKEN_FLOAT,
               TOKEN_STRING,
               TOKEN_LBRACE,
               TOKEN_RBRACE,
               TOKEN_LPAREN,
               TOKEN_RPAREN,
               TOKEN_INVALID };

  class Vertex {
  public:
    int weightIndex;
    int weightCount;
    Vec2f tc;
  };
  
  class Joint {
  public:
    std::string name;
    Vec3f pos;
    Quaternion quat;
    int parent;
    std::vector<int> children;
  };
  
  class Tri {
  public:
    int v[3]; // vertex indices
  };
  
  class Weight {
  public:
    int joint;
    float w;
	Vec3f pos;
  };
  
  class Mesh {
  public:
    std::string texture;
    std::vector<Vertex> verts;
    std::vector<Tri> tris;
    std::vector<Weight> weights;
  };
  
  class bFrame {
  public:
    std::vector<float> animatedComponents;
    std::vector<Joint> joints;
    Vec3f min; // min point of bounding box
    Vec3f max; // max point of bounding box
  };
  
  class JointInfo {
  public:
    std::string name;
    int parentIndex;
    int flags;
    int startIndex;
  };

  // stores data from an anim file
  class Anim {
  public:
    Anim(); 
    int numbFrames;
    int bFrameRate;
    int numAnimatedComponents;
    std::vector<bFrame> bFrames;
    std::vector<Joint> baseJoints;
    std::vector<JointInfo> jointInfo;
  };

class MD5Model {
public:
  MD5Model();
  ~MD5Model();

  void clear();
  void loadMesh(const char *filename);
  int  loadAnim(const char *filename);

  inline int getNumAnims() const { return int(anims.size()); }

  class Exception : public std::runtime_error {
  public:
    Exception(const std::string &msg): std::runtime_error(msg) { }
  };

//protected:
  

  // methods for parser
  void  readElements(std::ifstream &fin);
  void  readAnimElements(std::ifstream &fin, Anim &anim);
  void  readCommandLineEl(std::ifstream &fin);
  void  readNumMeshesEl(std::ifstream &fin);
  void  readNumJointsEl(std::ifstream &fin);
  void  readNumbFramesEl(std::ifstream &fin, Anim &anim);
  void  readbFrameRateEl(std::ifstream &fin, Anim &anim);
  void  readNumAnimatedComponentsEl(std::ifstream &fin, Anim &anim);
  void  readJointsEl(std::ifstream &fin);
  void  readMeshEl(std::ifstream &fin);
  void  readHierarchyEl(std::ifstream &fin, Anim &anim);
  void  readBoundsEl(std::ifstream &fin, Anim &anim);
  void  readBasebFrameEl(std::ifstream &fin, Anim &anim);
  void  readbFrameEl(std::ifstream &fin, Anim &anim);
  int   readInt(std::ifstream &fin);
  float readFloat(std::ifstream &fin);
  void  readVec(std::ifstream &fin, float *v, int n);

  // methods for lexer
  void readString(std::ifstream &fin, std::string &str);
  void skipComments(std::ifstream &fin);
  void skipWhitespace(std::ifstream &fin);
  TOKEN getNextToken(std::ifstream &fin, std::string *tokStr=NULL);
  int numJoints;
  int numMeshes;
  std::vector<Mesh*> meshes;
  std::vector<Joint> joints;
  std::vector<Anim*> anims;
};