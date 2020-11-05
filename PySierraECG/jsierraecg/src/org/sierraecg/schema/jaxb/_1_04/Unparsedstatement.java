//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.2.8-b130911.1802 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2014.08.15 at 08:44:39 PM EDT 
//


package org.sierraecg.schema.jaxb._1_04;

import java.math.BigInteger;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
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
 *         &lt;element name="lhsstatement" type="{http://www.w3.org/2001/XMLSchema}string"/>
 *         &lt;element name="rhsstatement" type="{http://www.w3.org/2001/XMLSchema}string"/>
 *       &lt;/sequence>
 *       &lt;attribute name="statementnumber" use="required" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" />
 *       &lt;attribute name="code" use="required" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="format" use="required" type="{http://www3.medical.philips.com}TYPEformat" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "", propOrder = {
    "lhsstatement",
    "rhsstatement"
})
@XmlRootElement(name = "unparsedstatement")
public class Unparsedstatement {

    @XmlElement(required = true)
    protected String lhsstatement;
    @XmlElement(required = true)
    protected String rhsstatement;
    @XmlAttribute(name = "statementnumber", required = true)
    @XmlSchemaType(name = "nonNegativeInteger")
    protected BigInteger statementnumber;
    @XmlAttribute(name = "code", required = true)
    protected String code;
    @XmlAttribute(name = "format", required = true)
    protected TYPEformat format;

    /**
     * Gets the value of the lhsstatement property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getLhsstatement() {
        return lhsstatement;
    }

    /**
     * Sets the value of the lhsstatement property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setLhsstatement(String value) {
        this.lhsstatement = value;
    }

    /**
     * Gets the value of the rhsstatement property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getRhsstatement() {
        return rhsstatement;
    }

    /**
     * Sets the value of the rhsstatement property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setRhsstatement(String value) {
        this.rhsstatement = value;
    }

    /**
     * Gets the value of the statementnumber property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getStatementnumber() {
        return statementnumber;
    }

    /**
     * Sets the value of the statementnumber property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setStatementnumber(BigInteger value) {
        this.statementnumber = value;
    }

    /**
     * Gets the value of the code property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getCode() {
        return code;
    }

    /**
     * Sets the value of the code property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setCode(String value) {
        this.code = value;
    }

    /**
     * Gets the value of the format property.
     * 
     * @return
     *     possible object is
     *     {@link TYPEformat }
     *     
     */
    public TYPEformat getFormat() {
        return format;
    }

    /**
     * Sets the value of the format property.
     * 
     * @param value
     *     allowed object is
     *     {@link TYPEformat }
     *     
     */
    public void setFormat(TYPEformat value) {
        this.format = value;
    }

}
