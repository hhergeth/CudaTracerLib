#include "StdAfx.h"
#include "MD5Parser.h"
#include <boost\algorithm\string.hpp>

// MD5 Loader, by A.J. Tavakoli



#include <iostream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstddef>
#include <cctype>

#define IS_WHITESPACE(c) (' ' == c || '\t' == c || '\r' ==c || '\n' == c )


MD5Model::MD5Model()
	: numJoints(0), numMeshes(0)
{

}

MD5Model::~MD5Model() {
  clear();
}


void MD5Model::clear() {
  // delete meshes
  for ( size_t i=0; i < meshes.size(); i++ )
	if ( meshes[i] )
	  delete meshes[i];

  // delete animations
  for ( size_t i=0; i < anims.size(); i++ )
	if ( anims[i] )
	  delete anims[i];

  meshes.clear();
  anims.clear();
  joints.clear();
}


void MD5Model::loadMesh(const char *filename) {
  // sanity check
  if ( !filename )
	throw Exception("MD5Model::loadMesh(): filename is NULL");

  // attempt to open file for reading
  std::ifstream fin(filename, std::ifstream::in);

  // was open successful?
  if ( !fin.is_open() ) {
	std::string msg = std::string("MD5Model::loadMesh(): unable to open ") +
					  std::string(filename) + std::string(" for reading");
	throw Exception(msg);
  }

  // read in file version
  std::string str;
  getNextToken(fin, &str);

  // token must read "MD5Version"
  if ( str != "MD5Version" )
	throw Exception("MD5Model::loadMesh(): expected 'MD5Version'");

  // get version #
  int ver = readInt(fin);

  // must be version 10
  if ( ver != 10 )
	throw Exception("MD5Model::loadMesh(): MD5Version must be 10");

  // clear any data before reading file
  clear();

  // read in all of the MD5Model elements
  readElements(fin);

  // close input file (should be done destructor anyway...)
  fin.close();
}


int MD5Model::loadAnim(const char *filename) {
  // attempt to open file for reading
  std::ifstream fin(filename, std::ifstream::in);

  if ( !fin.is_open() ) {
	std::string msg = std::string("MD5Model::loadAnim(): unable to open ") +
					  std::string(filename) + std::string(" for reading");
	throw Exception(msg);
  }

  // read in file version
  std::string str;
  getNextToken(fin, &str);

  // token must read "MD5Version"
  if ( str != "MD5Version" )
	throw Exception("MD5Model::loadAnim(): expected 'MD5Version'");
  
  // get version #
  int ver = readInt(fin);

  // must be version 10
  if ( ver != 10 )
	throw Exception("MD5Model::loadAnim(): MD5Version must be 10");

  Anim *anim = new Anim;
  if ( !anim )
	throw Exception("MD5Model::loadAnim(): could not allocate storage for animation");

  readAnimElements(fin, *anim);

  // close file (should be done by destructor anyway)
  fin.close();

  // add to array of animations
  int animIndex = (int)anims.size();
  anims.push_back(anim);


	for ( int i=0; i < anim->numbFrames; i++ )
	{
		// allocate storage for joints for this bFrame
		anim->bFrames[i].joints.resize(numJoints);

		for ( int j=0; j < numJoints; j++ )
		{
			const Joint &joint = anim->baseJoints[j];
  
			float pos[3]    = { joint.pos[0],  joint.pos[1],  joint.pos[2]  };
			float orient[3] = { joint.quat[0], joint.quat[1], joint.quat[2] };
  
			int n = 0;
			for ( int k=0; k < 3; k++ )
				if ( anim->jointInfo[j].flags & (1 << k)  )
				{
					pos[k] = anim->bFrames[i].animatedComponents[anim->jointInfo[j].startIndex + n];
					n++;
				}

			for ( int k=0; k < 3; k++ )
				if ( anim->jointInfo[j].flags & (8 << k)  )
				{
					orient[k] = anim->bFrames[i].animatedComponents[anim->jointInfo[j].startIndex + n];
					n++;
				}

			Quaternion q(0.5f, 0.5f, 0.5000001f, -0.5000001f);
			Vec3f pos2 = Vec3f(pos[0], pos[1], pos[2]);
			Quaternion oriented(orient[0], orient[1], orient[2]);
			if(anim->jointInfo[j].parentIndex == -1)
			{
				pos2 = q * pos2;
				oriented = (oriented * q);
				oriented.normalize();
			}

			Joint &bFrameJoint = anim->bFrames[i].joints[j];
			bFrameJoint.name   = anim->jointInfo[j].name;
			bFrameJoint.parent = anim->jointInfo[j].parentIndex;
			bFrameJoint.pos[0] = pos2.x;
			bFrameJoint.pos[1] = pos2.y;
			bFrameJoint.pos[2] = pos2.z;
			bFrameJoint.quat = oriented;
		}
	}

  return animIndex;
}


void MD5Model::readElements(std::ifstream &fin) {
  while ( !fin.eof() ) {
	std::string str;
	TOKEN tok = getNextToken(fin, &str);
  
	// token is TOKEN_INVALID when end of file is reached
	if ( TOKEN_INVALID == tok )
	  break;
  
	if ( str == "commandline" )
	  readCommandLineEl(fin);
	else if ( str == "numJoints" )
	  readNumJointsEl(fin);
	else if ( str == "numMeshes" )
	  readNumMeshesEl(fin);
	else if ( str == "joints" )
	  readJointsEl(fin);
	else if ( str == "mesh" )
	  readMeshEl(fin);
	else {
	  // invalid element
	  throw Exception( std::string("MD5Model::readElements(): unknown element type '") + str + "'");
	}
  } // while ( not EOF )
}


void MD5Model::readAnimElements(std::ifstream &fin, Anim &anim) {
  while ( !fin.eof() ) {
	std::string str;
	TOKEN tok = getNextToken(fin, &str);
	boost::algorithm::to_lower(str);

	// token is TOKEN_INVALID when end of file is reached
	if ( TOKEN_INVALID == tok )
	  break;

	if ( str == "commandline" )
	  readCommandLineEl(fin);
	else if ( str == "numjoints" ) {
	  // just make sure that the number of joints is the same as the number
	  // specified in the mesh file
	  int n = readInt(fin);

	  if ( n != numJoints )
		throw Exception("MD5Model::readAnimElements(): anim file does not match mesh");
	}
	else if ( str == "nummeshes" ) {
	  // just make sure that the number of meshes is the same as the number
	  // specified in the mesh file
	  int n = readInt(fin);

	  if ( n != numMeshes )
		throw Exception("MD5Model::readAnimElements(): anim file does not match mesh");
	}
	else if ( str == "numframes" )
	  readNumbFramesEl(fin, anim);
	else if ( str == "framerate" )
	  readbFrameRateEl(fin, anim);
	else if ( str == "numanimatedcomponents" )
	  readNumAnimatedComponentsEl(fin, anim);
	else if ( str == "hierarchy" )
	  readHierarchyEl(fin, anim);
	else if ( str == "bounds" )
	  readBoundsEl(fin, anim);
	else if ( str == "baseframe" )
	  readBasebFrameEl(fin, anim);
	else if ( str == "frame" )
	  readbFrameEl(fin, anim);
	else {
	  // invalid element
	  throw Exception( std::string("MD5Model::readAnimElements(): unknown element type '") + str + "'");
	}
  }
}


void MD5Model::readCommandLineEl(std::ifstream &fin) {
  // commandline elements can be ignored, but make sure the syntax is correct
  if ( getNextToken(fin) != TOKEN_STRING )
	throw Exception("MD5Model::readCommandLineEl(): expected string");
}


void MD5Model::readNumJointsEl(std::ifstream &fin) {
  // if number of joints has already been set, can't set it again
  if ( numJoints != 0 )
	throw Exception("MD5Model::readNumJointsEl(): numJoints has already been set");

  // read in next token, should be an integer
  int n = readInt(fin);

  if ( n <= 0 )
	throw Exception("MD5Model::readNumJointsEl(): numJoints must be greater than 0");

  numJoints =  n;
  //joints.resize(numJoints);
}


void MD5Model::readNumMeshesEl(std::ifstream &fin) {
  // if number of meshes has already been set, can't set it again
  if ( numMeshes != 0 )
	throw Exception("MD5Model::readNumMeshesEl(): numMeshes has already been set");

  // read in next token, should be an integer
  int n = readInt(fin);

  if ( n <= 0 )
	throw Exception("MD5Model::readNumMeshesEl(): numMeshes must be greater than 0");

  numMeshes =  n;
  //meshes.resize(numMeshes);
}


void MD5Model::readNumbFramesEl(std::ifstream &fin, Anim &anim) {
  // if number of bFrames has already been set, can't set it again
  if ( anim.numbFrames != 0 )
	throw Exception("MD5Model::readNumbFramesEl(): numbFrames has already been set");

  // read in next token, should be an integer
  int n = readInt(fin);

  if ( n <= 0 )
	throw Exception("MD5Model::readNumbFramesEl(): numbFrames must be greater than 0");

  anim.numbFrames =  n;
  anim.bFrames.resize(n);
}


void MD5Model::readbFrameRateEl(std::ifstream &fin, Anim &anim) {
  // if bFramerate has already been set, can't set it again
  if ( anim.bFrameRate != 0 )
	throw Exception("MD5Model::readbFrameRateEl(): bFrameRate has already been set");

  // read in next token, should be an integer
  int n = readInt(fin);

  if ( n <= 0 )
	throw Exception("MD5Model::readbFrameRateEl(): bFrameRate must be a positive integer");

  anim.bFrameRate =  n;
}


void MD5Model::readNumAnimatedComponentsEl(std::ifstream &fin, Anim &anim) {
  // make sure parameter hasn't been set, as has been done with all of the
  // others
  if ( anim.numAnimatedComponents != 0 )
	throw Exception("MD5Model::readNumAnimatedComponentsEl(): numAnimatedComponents has already been set");

  // read in next token, should be an integer
  int n = readInt(fin);

  if ( n < 0 )
	throw Exception("MD5Model::readNumAnimatedComponentsEl(): numAnimatedComponents must be a positive integer");

  anim.numAnimatedComponents = n;
}


void MD5Model::readJointsEl(std::ifstream &fin) {
  // make sure numJoints has been set
  if ( numJoints <= 0 )
	throw Exception("MD5Model::readJointsEl(): numJoints needs to be set before 'joints' block");

  TOKEN t = getNextToken(fin);

  // expect an opening brace { to begin block of joints
  if ( t != TOKEN_LBRACE )
	throw Exception("MD5Model::readJointsEl(): expected { to follow 'joints'");

  // read in each joint in block until '}' is hit
  // (if EOF is reached before this, then the read*() methods
  //  will close the file and throw an exception)
  std::string str;
  t = getNextToken(fin, &str);
  while ( t != TOKEN_RBRACE ) {
	Joint joint;

	// token should be name of joint (a string)
	if ( t != TOKEN_STRING )
	  throw Exception("MD5Model::readJointsEl(): expected joint name (string)'");

	// read in index of parent joint
	int parentIndex = readInt(fin);
  
	// read in joint position
	readVec(fin, (float*)&joint.pos, 3);
  
	// read in first 3 components of quaternion (must compute 4th)
	float quat[4];
	readVec(fin, quat, 3);

	joint.quat = Quaternion(quat[0], quat[1], quat[2]);

	// add index to parent's list of child joints (if it has a parent,
	// root joints have parent index -1)
	if ( parentIndex >= 0 ) {
	  if ( size_t(parentIndex) >= joints.size() )
		throw Exception("MD5Model::readJointsEl(): parent index is invalid");

	  joints[parentIndex].children.push_back( int(joints.size()) );
	}

	// add joint to vector of joints
	joints.push_back(joint);

	// get next token
	t = getNextToken(fin, &str);
  }
}


void MD5Model::readMeshEl(std::ifstream &fin) {
  // make sure numMeshes has been set
  if ( numMeshes <= 0 )
	throw Exception("MD5Model::readMeshesEl(): numMeshes needs to be set before 'mesh' block");

  TOKEN t = getNextToken(fin);

  // expect an opening brace { to begin block of joints
  if ( t != TOKEN_LBRACE )
	throw Exception("MD5Model::readMeshEl(): expected { to follow 'mesh'");

  Mesh *mesh = new Mesh;

  // read in all mesh information in block until '}' is hit
  // (if EOF is reached before this, then the read*() methods
  //  will close the file and throw an exception)
  std::string str;
  t = getNextToken(fin, &str);
  while ( t != TOKEN_RBRACE ) {
	if ( str == "vert" ) {
	  // syntax: vert (vertex index) '(' (x) (y) (z) ')' (weight index) (weight count)
	  Vertex vert;

	  int index = readInt(fin);
	  readVec(fin, (float*)&vert.tc, 2);
	  vert.weightIndex = readInt(fin);
	  vert.weightCount = readInt(fin);

	  // make sure index is within bounds of vector of vertices
	  if ( size_t(index) >= mesh->verts.size() )
		throw Exception("MD5Model::readMeshEl(): vertex index >= numverts");

	  mesh->verts[index] = vert;
	}
	else if ( str == "tri" ) {
	  // syntax: tri (triangle index) (v0 index) (v1 index) (v2 index)
	  Tri tri;
	  
	  int index = readInt(fin);

	  // make sure index is within bounds of vector of triangles
	  if ( size_t(index) >= mesh->tris.size() )
		throw Exception("MD5Model::readMeshEl(): triangle index >= numtris");

	  tri.v[0] = readInt(fin);
	  tri.v[1] = readInt(fin);
	  tri.v[2] = readInt(fin);
	  mesh->tris[index] = tri;
	}
	else if ( str == "weight" ) {
	  Weight weight;

	  int weightIndex = readInt(fin);
	  weight.joint = readInt(fin);
	  weight.w = readFloat(fin);
	  readVec(fin, (float*)&weight.pos, 3);

	  if ( size_t(weightIndex) >= mesh->weights.size() )
		throw Exception("MD5Model::readMeshEl(): weight index >= numweights");

	  mesh->weights[weightIndex] = weight;
	}
	else if ( str == "shader" ) {
	  std::string shader;
	  if ( getNextToken(fin, &shader) != TOKEN_STRING )
		throw Exception("MD5Model::readMeshEl(): expected string to follow 'shader'");
	  mesh->texture = shader;
	}
	else if ( str == "numverts" ) {
	  if ( mesh->verts.size() > 0 )
		throw Exception("MD5Model::readMeshEl(): numverts has already been set");

	  int n = readInt(fin);

	  if ( n <= 0 )
		throw Exception("MD5Model::readMeshEl(): numverts must be greater than 0");

	  mesh->verts.resize(n);
	}
	else if ( str == "numtris" ) {
	  if ( mesh->tris.size() > 0 )
		throw Exception("MD5Model::readMeshEl(): numtris has already been set");

	  int n = readInt(fin);

	  if ( n <= 0 )
		throw Exception("MD5Model::readMeshEl(): numtris must be greater than 0");

	  mesh->tris.resize(n);
	}
	else if ( str == "numweights" ) {
	  if ( mesh->weights.size() > 0 )
		throw Exception("MD5Model::readMeshEl(): numweights has already been set");

	  int n = readInt(fin);

	  if ( n <= 0 )
		throw Exception("MD5Model::readMeshEl(): numweights must be greater than 0");

	  mesh->weights.resize(n);
	}

	// read next token
	t = getNextToken(fin, &str);
  }

  meshes.push_back(mesh);
}


// reads in hierarchy block from anim file
void MD5Model::readHierarchyEl(std::ifstream &fin, Anim &anim) {
  TOKEN t = getNextToken(fin);

  // expect an opening brace { to begin hierarchy block
  if ( t != TOKEN_LBRACE )
	throw Exception("MD5Model::readHierarchyEl(): expected { to follow 'hierarchy'");

  anim.jointInfo.clear();

  // read in each joint in block until '}' is hit
  // (if EOF is reached before this, then the read*() methods
  //  will close the file and throw an exception)
  std::string str;
  t = getNextToken(fin, &str);
  while ( t != TOKEN_RBRACE ) {
	JointInfo info;

	// token should be name of a joint (a string)
	if ( t != TOKEN_STRING )
	  throw Exception("MD5Model::readHierarchyEl(): expected name (string)");

	info.name        = str;
	info.parentIndex = readInt(fin);
	info.flags       = readInt(fin);
	info.startIndex  = readInt(fin);

	anim.jointInfo.push_back(info);
	
	// get next token
	t = getNextToken(fin, &str);
  }

  if ( int(anim.jointInfo.size()) != numJoints )
	throw Exception("MD5Model::readHierarchyEl(): number of entires in hierarchy block does not match numJoints");
}


// reads in bounds block from anim file
void MD5Model::readBoundsEl(std::ifstream &fin, Anim &anim) {
  TOKEN t = getNextToken(fin);

  // expect an opening brace { to begin block
  if ( t != TOKEN_LBRACE )
	throw Exception("MD5Model::readBoundsEl(): expected { to follow 'bounds'");

  if ( anim.numbFrames != int(anim.bFrames.size()) )
	throw Exception("MD5Model::readBoundsEl(): bFrames must be allocated before setting bounds");

  // read in bound for each bFrame
  for ( int i=0; i < anim.numbFrames; i++ ) {
	  readVec(fin, (float*)&anim.bFrames[i].min, 3);
	  readVec(fin, (float*)&anim.bFrames[i].max, 3);
  }

  // next token must be a closing brace }
  t = getNextToken(fin);

  if ( TOKEN_LPAREN == t )
	throw Exception("MD5Model::readBoundsEl(): number of bounds exceeds number of bFrames");

  // expect a closing brace } to end block
  if ( t != TOKEN_RBRACE )
	throw Exception("MD5Model::readBoundsEl(): expected }");
}


void MD5Model::readBasebFrameEl(std::ifstream &fin, Anim &anim) {
  TOKEN t = getNextToken(fin);

  // expect an opening brace { to begin block
  if ( t != TOKEN_LBRACE )
	throw Exception("MD5Model::readBasebFrameEl(): expected { to follow 'basebFrame'");

  anim.baseJoints.resize(numJoints);

  int i;
  for ( i=0; i < numJoints; i++ ) {
	// read in joint position
	  readVec(fin, (float*)&anim.baseJoints[i].pos, 3);

	// read in first 3 components of quaternion (must compute 4th)
	float quat[3];
	readVec(fin, quat, 3);

	anim.baseJoints[i].quat = Quaternion(quat[0], quat[1], quat[2]);
  }

  if ( i < numJoints - 1 )
	throw Exception("MD5Model::readBasebFrameEl(): not enough joints");

  // next token must be a closing brace }
  t = getNextToken(fin);

  if ( TOKEN_LPAREN == t )
	throw Exception("MD5Model::readBasebFrameEl(): too many joints");

  // expect a closing brace } to end block
  if ( t != TOKEN_RBRACE )
	throw Exception("MD5Model::readBasebFrameEl(): expected }");
}


void MD5Model::readbFrameEl(std::ifstream &fin, Anim &anim) {
  // numAnimatedComponents has to have been set before bFrame element
  if ( 0 > anim.numAnimatedComponents )
	throw Exception("MD5Model::readbFrameEl(): numAnimatedComponents must be set before 'bFrame' block");

  // read bFrame index
  int bFrameIndex = readInt(fin);

  if ( bFrameIndex < 0 || bFrameIndex >= anim.numbFrames )
	throw Exception("MD5Model::readbFrameEl(): invalid bFrame index");

  // get reference to bFrame and set number of animated components
  bFrame &bFrame = anim.bFrames[bFrameIndex];
  bFrame.animatedComponents.resize(anim.numAnimatedComponents);

  TOKEN t = getNextToken(fin);

  // expect an opening brace { to begin block
  if ( t != TOKEN_LBRACE )
	throw Exception("MD5Model::readbFrameEl(): expected { to follow bFrame index");

  for ( int i=0; i < anim.numAnimatedComponents; i++ )
	bFrame.animatedComponents[i] = readFloat(fin);

  t = getNextToken(fin);

  // expect a closing brace } to end block
  if ( t != TOKEN_RBRACE )
	throw Exception("MD5Model::readbFrameEl(): expected }");
}


// reads in a string terminal and stores it in str
// (assumes opening " has already been read in)
void MD5Model::readString(std::ifstream &fin, std::string &str) {
  str = std::string();

  // read characters until closing " is found
  char c = '\0';
  do {
	fin.get(c);

	if ( fin.eof() )
	  throw Exception("MD5Model::readString(): reached end of file before \" was found");

	if ( c != '"')
	  str += c;
  } while ( c != '"' );
}


// reads in an integer terminal and returns its value
int MD5Model::readInt(std::ifstream &fin) {
  std::string str;
  TOKEN t = getNextToken(fin, &str);

  if ( t != TOKEN_INT )
	throw Exception("MD5Model::readInt(): expected integer");

  return atoi( str.c_str() );
}


// reads in a float terminal and returns its value
float MD5Model::readFloat(std::ifstream &fin) {
  std::string str;
  TOKEN t = getNextToken(fin, &str);

  // integer tokens are just numbers with out a decimal point, so they'll
  // suffice here as well
  if ( t != TOKEN_FLOAT && t != TOKEN_INT )
	throw Exception("MD5Model::readFloat(): expected float");

  float f = 0.0f;
  sscanf(str.c_str(), "%f", &f);

  return f;
}


// reads in sequence consisting of n floats enclosed by parentheses
void MD5Model::readVec(std::ifstream &fin, float *v, int n) {
  if ( getNextToken(fin) != TOKEN_LPAREN )
	throw Exception("MD5Model::readVec(): expected '('");

  for ( int i=0; i < n; i++ )
	v[i] = readFloat(fin);

  if ( getNextToken(fin) != TOKEN_RPAREN )
	throw Exception("MD5Model::readVec(): expected ')'");
}


void MD5Model::skipComments(std::ifstream &fin) {
  char c;
  fin.get(c);

  if ( c != '/' )
	throw Exception("MD5Model::skipComments(): invalid comment, expected //");

  while ( !fin.eof() && c != '\n' )
	fin.get(c);

  // put back last character read
  fin.putback(c);
}


// reads until first non-whitespace character
void MD5Model::skipWhitespace(std::ifstream &fin) {
  char c = '\0';
  while ( !fin.eof() ) {
	fin.get(c);

	if ( !IS_WHITESPACE(c) ) {
	  fin.putback(c);
	  break;
	}
  }
}


// reads in next token from file and matches it to a token type,
// if tokStr is non-NULL then it will be set to the text of the token
TOKEN MD5Model::getNextToken(std::ifstream &fin, std::string *tokStr) {
  skipWhitespace(fin);
  std::string str;

  TOKEN t = TOKEN_INVALID;

  while ( !fin.eof() ) {
	char c = '\0';
	fin.get(c);

	// single character tokens
	if ( '{' == c || '}' == c || '(' == c || ')' == c ) {
	  // if already reading in a token, treat this as a delimiter
	  if ( t != TOKEN_INVALID ) {
		fin.putback(c);
		if ( tokStr != NULL )
		  (*tokStr) = str;
	  }
 
	  if ( '{' == c )
		t = TOKEN_LBRACE;
	  if ( '}' == c )
		t = TOKEN_RBRACE;
	  if ( '(' == c )
		t = TOKEN_LPAREN;
	  if ( ')' == c )
		t = TOKEN_RPAREN;

	  if ( tokStr) {
		(*tokStr) = std::string();
		(*tokStr) += c;
	  }
	  return t;
	}
	if ( isdigit(c) ) {
	  str += c;
	  if ( TOKEN_INVALID == t )
		t = TOKEN_INT;
	  else if ( t != TOKEN_INT && t != TOKEN_FLOAT && t != TOKEN_KEYWORD ) {
		std::string msg("MD5Model::getNextToken(): invalid token '");
		msg += str + "'";
		throw Exception(msg);
	  }
	}
	if ( '-' == c ) {
	  str += c;
	  if ( TOKEN_INVALID == t )
		t = TOKEN_INT;
	  else {
		std::string msg("MD5Model::getNextToken(): invalid token '");
		msg += str + "'";
		throw Exception(msg);
	  }
	}
	if ( isalpha(c) ) {
	  str += c;
	  if ( TOKEN_INVALID == t )
		t = TOKEN_KEYWORD;
	  else if ( t != TOKEN_KEYWORD ) {
		std::string msg("MD5Model::getNextToken(): invalid token '");
		msg += str + "'";
		throw Exception(msg);
	  }
	}
	if ( '"' == c ) {
	  // treat as a delimeter if already reading in a token
	  if ( t != TOKEN_INVALID ) {
		fin.putback(c);
		if ( tokStr != NULL )
		  (*tokStr) = str;
		return t;
	  }
	  readString(fin, str);

	  if ( tokStr != NULL )
		(*tokStr) = str;

	  return TOKEN_STRING;
	}
	if ( '.' == c ) {
	  str += c;
	  if ( t != TOKEN_INT ) {
		std::string msg("MD5Model::getNextToken(): invalid token '");
		msg += str + "'";
		throw Exception(msg);
	  }
	  t = TOKEN_FLOAT;
	}
	if ( '/' == c ) {
	  // treat as a delimeter if already reading in a token
	  if ( t != TOKEN_INVALID ) {
		if ( tokStr != NULL )
		  (*tokStr) = str;
		return t;
	  }
	  
	  skipComments(fin);
	  skipWhitespace(fin);
	  continue;
	}

	// treat whitespace as a delimeter
	if ( IS_WHITESPACE(c) ) {
	  if ( tokStr != NULL )
		(*tokStr) = str;
	  return t;
	}

	// at this point token type should be set, if it hasn't been then
	// token is invalid
	if ( TOKEN_INVALID == t ) {
		std::string msg("MD5Model::getNextToken(): invalid token '");
		str += c;
		msg += str + "'";
		throw Exception(msg);
	}
  }

  return TOKEN_INVALID;
}


Anim::Anim():
	 numbFrames(0),
	 bFrameRate(0),
	 numAnimatedComponents(0) {

}

