

/* First created by JCasGen Wed Feb 20 11:47:50 CET 2019 */
package org.hucompute.services.type.biocreative;

import org.apache.uima.jcas.JCas; 
import org.apache.uima.jcas.JCasRegistry;
import org.apache.uima.jcas.cas.TOP_Type;

import org.apache.uima.jcas.tcas.Annotation;


/** 
 * Updated by JCasGen Wed Feb 20 11:47:50 CET 2019
 * XML source: /home/ahemati/workspaceGit/LSTMVoter/src/main/resources/desc/type/BioCreative.xml
 * @generated */
public class Abstract extends Annotation {
  /** @generated
   * @ordered 
   */
  @SuppressWarnings ("hiding")
  public final static int typeIndexID = JCasRegistry.register(Abstract.class);
  /** @generated
   * @ordered 
   */
  @SuppressWarnings ("hiding")
  public final static int type = typeIndexID;
  /** @generated
   * @return index of the type  
   */
  @Override
  public              int getTypeIndexID() {return typeIndexID;}
 
  /** Never called.  Disable default constructor
   * @generated */
  protected Abstract() {/* intentionally empty block */}
    
  /** Internal - constructor used by generator 
   * @generated
   * @param addr low level Feature Structure reference
   * @param type the type of this Feature Structure 
   */
  public Abstract(int addr, TOP_Type type) {
    super(addr, type);
    readObject();
  }
  
  /** @generated
   * @param jcas JCas to which this Feature Structure belongs 
   */
  public Abstract(JCas jcas) {
    super(jcas);
    readObject();   
  } 

  /** @generated
   * @param jcas JCas to which this Feature Structure belongs
   * @param begin offset to the begin spot in the SofA
   * @param end offset to the end spot in the SofA 
  */  
  public Abstract(JCas jcas, int begin, int end) {
    super(jcas);
    setBegin(begin);
    setEnd(end);
    readObject();
  }   

  /** 
   * <!-- begin-user-doc -->
   * Write your own initialization here
   * <!-- end-user-doc -->
   *
   * @generated modifiable 
   */
  private void readObject() {/*default - does nothing empty block */}
     
 
    
  //*--------------*
  //* Feature: id

  /** getter for id - gets 
   * @generated
   * @return value of the feature 
   */
  public String getId() {
    if (Abstract_Type.featOkTst && ((Abstract_Type)jcasType).casFeat_id == null)
      jcasType.jcas.throwFeatMissing("id", "org.hucompute.services.type.biocreative.Abstract");
    return jcasType.ll_cas.ll_getStringValue(addr, ((Abstract_Type)jcasType).casFeatCode_id);}
    
  /** setter for id - sets  
   * @generated
   * @param v value to set into the feature 
   */
  public void setId(String v) {
    if (Abstract_Type.featOkTst && ((Abstract_Type)jcasType).casFeat_id == null)
      jcasType.jcas.throwFeatMissing("id", "org.hucompute.services.type.biocreative.Abstract");
    jcasType.ll_cas.ll_setStringValue(addr, ((Abstract_Type)jcasType).casFeatCode_id, v);}    
  }

    