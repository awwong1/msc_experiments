//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.2.8-b130911.1802 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2014.08.15 at 08:44:37 PM EDT 
//


package org.sierraecg.schema.jaxb._1_03;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.XmlSchemaType;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for anonymous complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType>
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element ref="{http://www3.medical.philips.com}patientid"/>
 *         &lt;element ref="{http://www3.medical.philips.com}viperuniquepatientid" minOccurs="0"/>
 *         &lt;element ref="{http://www3.medical.philips.com}name"/>
 *         &lt;element ref="{http://www3.medical.philips.com}age"/>
 *         &lt;element name="pacestatus" type="{http://www3.medical.philips.com}TYPEpacestatus"/>
 *         &lt;element ref="{http://www3.medical.philips.com}sex"/>
 *         &lt;element ref="{http://www3.medical.philips.com}race" minOccurs="0"/>
 *         &lt;element ref="{http://www3.medical.philips.com}height" minOccurs="0"/>
 *         &lt;element ref="{http://www3.medical.philips.com}weight" minOccurs="0"/>
 *       &lt;/sequence>
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "", propOrder = {
    "patientid",
    "viperuniquepatientid",
    "name",
    "age",
    "pacestatus",
    "sex",
    "race",
    "height",
    "weight"
})
@XmlRootElement(name = "generalpatientdata")
public class Generalpatientdata {

    @XmlElement(required = true)
    protected String patientid;
    protected String viperuniquepatientid;
    @XmlElement(required = true)
    protected Name name;
    @XmlElement(required = true)
    protected Age age;
    @XmlElement(required = true)
    @XmlSchemaType(name = "string")
    protected TYPEpacestatus pacestatus;
    @XmlElement(required = true)
    @XmlSchemaType(name = "string")
    protected TYPEsex sex;
    protected Race race;
    protected Height height;
    protected Weight weight;

    /**
     * Gets the value of the patientid property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getPatientid() {
        return patientid;
    }

    /**
     * Sets the value of the patientid property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setPatientid(String value) {
        this.patientid = value;
    }

    /**
     * Gets the value of the viperuniquepatientid property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getViperuniquepatientid() {
        return viperuniquepatientid;
    }

    /**
     * Sets the value of the viperuniquepatientid property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setViperuniquepatientid(String value) {
        this.viperuniquepatientid = value;
    }

    /**
     * Gets the value of the name property.
     * 
     * @return
     *     possible object is
     *     {@link Name }
     *     
     */
    public Name getName() {
        return name;
    }

    /**
     * Sets the value of the name property.
     * 
     * @param value
     *     allowed object is
     *     {@link Name }
     *     
     */
    public void setName(Name value) {
        this.name = value;
    }

    /**
     * Gets the value of the age property.
     * 
     * @return
     *     possible object is
     *     {@link Age }
     *     
     */
    public Age getAge() {
        return age;
    }

    /**
     * Sets the value of the age property.
     * 
     * @param value
     *     allowed object is
     *     {@link Age }
     *     
     */
    public void setAge(Age value) {
        this.age = value;
    }

    /**
     * Gets the value of the pacestatus property.
     * 
     * @return
     *     possible object is
     *     {@link TYPEpacestatus }
     *     
     */
    public TYPEpacestatus getPacestatus() {
        return pacestatus;
    }

    /**
     * Sets the value of the pacestatus property.
     * 
     * @param value
     *     allowed object is
     *     {@link TYPEpacestatus }
     *     
     */
    public void setPacestatus(TYPEpacestatus value) {
        this.pacestatus = value;
    }

    /**
     * Gets the value of the sex property.
     * 
     * @return
     *     possible object is
     *     {@link TYPEsex }
     *     
     */
    public TYPEsex getSex() {
        return sex;
    }

    /**
     * Sets the value of the sex property.
     * 
     * @param value
     *     allowed object is
     *     {@link TYPEsex }
     *     
     */
    public void setSex(TYPEsex value) {
        this.sex = value;
    }

    /**
     * Gets the value of the race property.
     * 
     * @return
     *     possible object is
     *     {@link Race }
     *     
     */
    public Race getRace() {
        return race;
    }

    /**
     * Sets the value of the race property.
     * 
     * @param value
     *     allowed object is
     *     {@link Race }
     *     
     */
    public void setRace(Race value) {
        this.race = value;
    }

    /**
     * Gets the value of the height property.
     * 
     * @return
     *     possible object is
     *     {@link Height }
     *     
     */
    public Height getHeight() {
        return height;
    }

    /**
     * Sets the value of the height property.
     * 
     * @param value
     *     allowed object is
     *     {@link Height }
     *     
     */
    public void setHeight(Height value) {
        this.height = value;
    }

    /**
     * Gets the value of the weight property.
     * 
     * @return
     *     possible object is
     *     {@link Weight }
     *     
     */
    public Weight getWeight() {
        return weight;
    }

    /**
     * Sets the value of the weight property.
     * 
     * @param value
     *     allowed object is
     *     {@link Weight }
     *     
     */
    public void setWeight(Weight value) {
        this.weight = value;
    }

}
